import argparse

class TrainParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        #unet arguments
        self.parser.add_argument('--lr_unet', default=2, type=float,help="learning rate of unet")
        self.parser.add_argument("--unet_ngf", default=64, type=int, help="unet base channel dimension")
        self.parser.add_argument("--unet_input_nc", type=int, default=3, help="input spectrogram number of channels")
        self.parser.add_argument("--unet_output_nc", type=int, default=3, help="output spectrogram number of channels")
        #optimizer arguments
        self.parser.add_argument("--audioVisual_feature_dim", type=int, default=1280, help="dimension of audioVisual feature map")
        self.parser.add_argument("--identity_feature_dim", type=int, default=64, help="dimension of identity feature map")
        self.parser.add_argument("--weights_unet", type=str, default="", help="weights for unet")
        self.parser.add_argument("--beta1", default=0.9, type=float, help="momentum for sgd, beta1 for adam")
        self.parser.add_argument("--weight_decay", default=0.0001, type=float, help="weight regularizer")
        self.initialized = True
        
        self.parser.add_argument("--batchSize", default=32, type=int, help="input batch size")
        self.parser.add_argument("--epoch_count", type=int, default=0, help="")
        self.parser.add_argument("--niter", default=1, help="")
        self.parser.add_argument("--n_fft", default=512, type=int, help="stft hop length")
        self.parser.add_argument("--hop_size", default=160, type=int, help="stft hop length")
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        return self.opt