import torch
import copy

class StreamTensorMemory(object):
    def __init__(self, batch_size, use_overlap=False):
        self.train_bs = batch_size
        self.training = True
        self.bs = self.train_bs
        self.use_overlap = use_overlap

        self.train_memory_list = [None for i in range(self.bs)]
        self.train_img_metas_memory = [None for i in range(self.bs)]
        if self.use_overlap:
            self.train_overlap_memory_list = [None for i in range(self.bs)]
            self.test_overlap_memory_list = [None] # bs = 1 when testing

        self.test_memory_list = [None] # bs = 1 when testing
        self.test_img_metas_memory = [None]
    
    @property
    def memory_list(self):
        if self.training:
            return self.train_memory_list
        else:
            return self.test_memory_list
    
    @property
    def img_metas_memory(self):
        if self.training:
            return self.train_img_metas_memory
        else:
            return self.test_img_metas_memory

    def update(self, memory, img_metas, overlap_memory=None):
        for i in range(self.bs):
            self.memory_list[i] = memory[i].clone().detach()
            self.img_metas_memory[i] = copy.deepcopy(img_metas[i])
            if self.use_overlap:
                if self.training:
                    self.train_overlap_memory_list[i] = overlap_memory[i].clone().detach()
                else:
                    self.test_overlap_memory_list[0] = overlap_memory[0].clone().detach()
        
    def reset_single(self, idx):
        self.memory_list[idx] = None
        self.img_metas_memory[idx] = None
        if self.use_overlap:
            if self.training:
                self.train_overlap_memory_list[idx] = None
            else:
                self.test_overlap_memory_list[0] = None

    def get(self, img_metas):
        '''
        img_metas: list[img_metas]
        '''

        tensor_list = []
        img_metas_list = []
        is_first_frame_list = []
        if self.use_overlap:
            overlap_list = []
        
        for i in range(self.bs):
            if not self.img_metas_memory[i]:
                is_first_frame = True
            else:
                is_first_frame = (img_metas[i]['scene_name'] != self.img_metas_memory[i]['scene_name'])

            if is_first_frame:
                self.reset_single(i)

            tensor_list.append(self.memory_list[i])
            img_metas_list.append(self.img_metas_memory[i])
            is_first_frame_list.append(is_first_frame)
            if self.use_overlap:
                if self.training:
                    overlap_list.append(self.train_overlap_memory_list[i])
                else:
                    assert i == 0
                    overlap_list.append(self.test_overlap_memory_list[0])

        result = {
            'tensor': tensor_list,
            'img_metas': img_metas_list,
            'is_first_frame': is_first_frame_list,
        }
        if self.use_overlap:
            result['overlap'] = overlap_list
        
        return result
    
    def train(self, mode=True):
        self.training = mode
        if mode:
            self.bs = self.train_bs
        else:
            self.bs = 1

    def eval(self):
        self.train(False)
