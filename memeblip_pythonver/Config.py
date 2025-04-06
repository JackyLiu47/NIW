import os

class Config:
    def __init__(self):
        # 路径设置
        self.root_dir = '../memeblip'
        self.img_folder = os.path.join(self.root_dir, 'PrideMM/Images')
        self.info_file = os.path.join(self.root_dir, 'PrideMM/PrideMM.csv')
        self.checkpoint_path = os.path.join(self.root_dir, 'checkpoints')
        self.checkpoint_file = os.path.join(self.checkpoint_path, 'model.ckpt')
        self.trainData_path = '../cached_features/train.pt'
        self.valData_path = '../cached_features/val.pt'

        # 模型与数据集设置
        self.clip_variant = "ViT-L/14"
        self.dataset_name = 'Pride'
        self.name = 'MemeBLIP'
        self.label = 'hate'
        self.seed = 42
        self.test_only = False
        self.device = 'cuda'
        self.gpus = [0]

        # 根据任务类型动态设置类别
        if self.label == 'hate':
            self.class_names = ['Benign Meme', 'Harmful Meme']
        elif self.label == 'humour':
            self.class_names = ['No Humour', 'Humour']
        elif self.label == 'target':
            self.class_names = ['No particular target', 'Individual', 'Community', 'Organization']
        elif self.label == 'stance':
            self.class_names = ['Neutral', 'Support', 'Oppose']

        # 超参数设置
        self.batch_size = 64
        self.image_size = 224
        self.num_mapping_layers = 1
        self.unmapped_dim = 768
        self.map_dim = 1024
        self.num_pre_output_layers = 2
        self.drop_probs = [0.4, 0.2, 0.3]
        self.dropout_rate = 0.5
        self.hidden_dim = 1024
        self.lr = 5e-5
        self.max_epochs = 32
        self.weight_decay = 1e-4
        self.num_classes = len(self.class_names)
        self.scale = 30
        self.print_model = True
        self.fast_process = True
        self.reproduce = False
        self.ratio = 0.7
        self.num_layers = 3
        self.activation = 'ReLU'
        self.hidden_dim1 = 1024