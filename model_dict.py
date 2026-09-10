from models import PINN, QRes, FLS, KAN, PINNsFormer, PINNsFormer_Enc_Only, PINNMamba


def get_model(args):
    model_dict = {
        'PINN': PINN,
        'QRes': QRes,
        'FLS': FLS,
        'KAN': KAN,
        'PINNsFormer': PINNsFormer,
        'PINNMamba': PINNMamba,
        'PINNsFormer_Enc_Only': PINNsFormer_Enc_Only,
    }
    return model_dict[args.model]