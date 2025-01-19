import math


def get_noise_from_batchsize(B, ref_noise, ref_B, beta=1):
    """
    output the noise necessary to keep our "physical constant" eta constant.
    """
    return ref_noise / ((ref_B / B) * beta)


def get_epochs_from_batchsize(B, ref_nb_steps, size_dataset):
    """
    output the approximate number of epochs necessary to keep our "physical constant" eta constant.
    We use a ceil, but please not that the last epoch will stop when we reach 'ref_nb_steps' steps.
    """
    return math.ceil(ref_nb_steps * B / size_dataset)
