import os
import torch
import random
import numpy as np
import copy
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import AlproBaseDataset
from src.datasets.randaugment import TemporalConsistentRandomAugment


class AlproVideoQADataset(AlproBaseDataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """

    ##@ add ovqa config
    open_ended_qa_names = ["frameqa", "msrvtt_qa", "msvd_qa", "ovqa"]

    def __init__(self, task_type, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, ans2label=None,
                 ensemble_n_clips=1, return_label=True, is_train=False, random_sample_clips=True, 
                 video_fmt='.mp4', img_db_type='lmdb', 
                 ##@ new init parameters
                 ret_model=None, c_level=None, f_level=None):
        super(AlproVideoQADataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, img_db_type=img_db_type,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.return_label = return_label
        self.is_train = is_train
        self.task_type = task_type
        self.ans2label = ans2label
        self.num_labels = len(ans2label)
        self.random_sample_clips = random_sample_clips
        self.label2ans = {v: k for k, v in ans2label.items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

        self.video_fmt = video_fmt

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

        ##@ set ret_model & hyperparameter
        self.ret_model = ret_model
        self.c_level = c_level
        self.f_level = f_level


    def __len__(self):
        return len(self.datalist)


    def __getitem__(self, index):
        # skip error videos:
        num_retries = 5
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            if self.ensemble_n_clips > 1:
                raise NotImplementedError('Do not support multiple clips for now.')
            else:
                video_path = os.path.join(self.img_db_dir, vid_id + self.video_fmt) 
                ###############################################################################################################
                # @ This is where the Online Fibonacci algorithm to be implemented
                ###############################################################################################################
                # BASELINE
                # FurtherTODO
                # vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)
                
                # ONLINE
                # TOTEST
                #print(video_path)
                
                """
                frame_indices, vid_frm_array = self.online_fib_matching_frames_decord(video_path = video_path, 
                                                                       text = examples[0]["question"], # 1 example per video
                                                                       ret_model = self.ret_model, 
                                                                       c_level = self.c_level, 
                                                                       fib_level = self.f_level)
                """
                
                frame_indices, vid_frm_array = self.online_one_by_one_matching_frames_decord(video_path = video_path, 
                                                                       text = examples[0]["question"], # 1 example per video
                                                                       ret_model = self.ret_model, 
                                                                       c_level = self.c_level, 
                                                                       fib_level = self.f_level)

                # Select a random video if the current video was not able to access.
                if vid_frm_array is None:
                    LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                                f"Will randomly sample an example as a replacement.")
                    index = random.randint(0, len(self) - 1)
                    continue 

                # list of int, (bz=1, T, C, H, W)
                vid_frm_array = vid_frm_array.squeeze(0) #(T, C, H, W)
                vid_frm_array = vid_frm_array.cpu()

                ##@ debug check shape
                #print(vid_frm_array.shape)

                ##@ debug nomarlization
                #print(f'Test nomarlization{torch.max(vid_frm_array)}' )
            if self.randaug:
                # Double check random augment
                # TOTEST
                vid_frm_array = self.randaug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            examples = [self._get_single_example(e) for e in examples]
            return dict(
                vid=vid_frm_array,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"]
        )
        if self.task_type in self.open_ended_qa_names:
            if self.return_label:
                example["label"] = self.ans2label[example["label"]]
        if not self.return_label:
            example["label"] = None
        return example
    
    # The evaluate function needs to be re-designed
    # TODO
    def evaluate_qa(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "question_id": int,
                    "answer": int or float, either answer_idx (int)
                }
        Returns:
            TGIF-QA score
        """
        preds = []
        gts = []
        # for frameQA
        answer_types = []
        ##@ add ovqa config
        answer_type2idx = dict(
            frameqa={"object": 0, "number": 1, "color": 2, "location": 3},
            msrvtt_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
            msvd_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
            ##@ add ovqa config
            ovqa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
        )

        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
        if self.task_type in self.open_ended_qa_names:  # convert ans_idx, int --> str
            qid2pred_ans = {k: self.label2ans[v] for k, v in qid2pred_ans.items()}

        for qid, pred_ans in qid2pred_ans.items():
            preds.append(pred_ans)

            gt_data = self.qid2data[qid]
            gt_ans = gt_data["answer"]
            if self.task_type in self.open_ended_qa_names:
                answer_types.append(answer_type2idx[self.task_type][gt_data["answer_type"]])
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        metrics["overall_acc"] = float(np.mean(preds == gts))
        if self.task_type in self.open_ended_qa_names:
            answer_types = np.array(answer_types)
            ratios = dict()
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                answer_type_mask = answer_types == ans_type_idx
                answer_type_corrects = (
                        preds[answer_type_mask] == gts[answer_type_mask])
                metrics[f"{ans_type}_acc"] = float(
                    np.mean(answer_type_corrects)) if len(answer_type_corrects) != 0 else 0
                ratios[f"{ans_type}_ratio"] = [
                    1. * len(answer_type_corrects) / len(answer_types),
                    len(answer_type_corrects)]
            metrics["ratios"] = ratios
        return metrics


class VideoQACollator(object):
    def __init__(self, tokenizer, max_length=20, task_type="action", n_options=5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.n_options = n_options

    def collate_batch(self, batch):
        v_collate = default_collate

        ##@ This v_collate function needs to be re-considerate for     
        # Original
        # visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, C, H, W)
        # Changed
        # visual_inputs T is not same for each example
        visual_inputs = [d["vid"] for d in batch] # list of size (B, ), element: (T, C, H, W)
        
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in ["action", "transition"]:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )  # (B * n_options, )
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        return dict(
            visual_inputs=visual_inputs,  # Original: (B, #frm, H, W, C) -> #Current: list of size B, element: (T, C, H, W)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,
            labels=labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
