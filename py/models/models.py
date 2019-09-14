# TODO: cleanup 
from models.models_zoo import Res34Unetv4

def get_model(network):
    if network == 'Res34Unetv3':
        model = Res34Unetv3()
        return model 
    elif network == 'Res34Unetv4':
        model = Res34Unetv4()
        return model 
    elif network == 'Res34Unetv5':
        model = Res34Unetv5()
        return model 
    else:
        raise ValueError('Unknown network ' + network)

    return model
