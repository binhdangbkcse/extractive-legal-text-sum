class VietnewsConfig:
    def __init__(self, train_data_dir, val_data_dir, test_data_dir, bert_name, MAX_SEQ_LEN, MAX_DOC_LEN, bert_hidden, bert_n_layers, windows_size, out_channels, lstm_hidden, device, batch_size, num_epochs, warmup_steps, gradient_accumulation_steps, print_freq, save_dir):
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.bert_name = bert_name  # my Roberta pretrained model on  4gb of text

        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.MAX_DOC_LEN = MAX_DOC_LEN

        self.bert_hidden = bert_hidden
        self.bert_n_layers = bert_n_layers

        self.windows_size = windows_size
        self.out_channels = out_channels
        self.lstm_hidden = lstm_hidden
        self.device = device

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.print_freq = print_freq
        self.save_dir = save_dir

# config = VietnewsConfig()
# config.__dict__