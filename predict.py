import torch
import argparse, pprint

from train_tools import *
from SetupDict import MODELS, PREDICTOR

# Set torch base print precision
torch.set_printoptions(6)


def main(args):
    """Execute prediction and save the results"""

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model_args = args.pred_setups.model
    model = MODELS[model_args.name](**model_args.params)

    if "ensemble" in args.pred_setups.name:
        #weights = torch.load(args.pred_setups.model_path1, map_location="cpu")
        
        weights = torch.load(args.pred_setups.model_path1, map_location=device)
        model.load_state_dict(weights, strict=False)
        ###
        model = model.to(device)

        model_aux = MODELS[model_args.name](**model_args.params)
        weights_aux = torch.load(args.pred_setups.model_path2, map_location=device)
        model_aux.load_state_dict(weights_aux, strict=False)

        ###
        model_aux = model_aux.to(device)

        predictor = PREDICTOR[args.pred_setups.name](
            model,
            model_aux,
            device,
            args.pred_setups.input_path,
            args.pred_setups.output_path,
            args.pred_setups.make_submission,
            args.pred_setups.exp_name,
            args.pred_setups.algo_params,
        )

    else:
        weights = torch.load(args.pred_setups.model_path, map_location=device)
        model.load_state_dict(weights, strict=False)

        ###
        model = model.to(device)

        predictor = PREDICTOR[args.pred_setups.name](
            model,
            device,
            args.pred_setups.input_path,
            args.pred_setups.output_path,
            args.pred_setups.make_submission,
            args.pred_setups.exp_name,
            args.pred_setups.algo_params,
        )

    _ = predictor.conduct_prediction()


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Config file processing")
parser.add_argument(
    "--config_path", default="./config/step3_prediction/base_prediction.json", type=str
)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

   # Print configuration dictionary pretty
    pprint_config(opt)

    # Run experiment
    main(opt)
