import argparse
import os

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Text-grounded SCD training options"
        )
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--epoch", type=int, default=200, help="Total training epochs")
        self.parser.add_argument("--base_lr", type=float, default=0.0001, help="Base learning rate for pre-trained backbone")
        self.parser.add_argument("--head_lr_mult", type=float, default=10.0, help="Learning rate multiplier for S2CE, DTSA, and CSGD task modules")
        self.parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
        self.parser.add_argument("--config", type=str, default="configs/SECOND.json", help="Path to dataset config json")
        self.parser.add_argument(
            "--sam3_path",
            type=str,
            default="model_weights/sam3.pt",
            help="Path to SAM 3 pretrained weights",
        )
        self.parser.add_argument(
            "--save_path",
            type=str,
            default="checkpoints/text_grounded_scd",
            help="Path to save checkpoints",
        )
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()
        print("=" * 20 + " Options " + "=" * 20)
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("=" * 47)
        os.makedirs(args.save_path, exist_ok=True)
        return args
