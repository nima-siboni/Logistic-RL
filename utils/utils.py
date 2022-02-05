"""
Some utilities
"""
import sys


def conf_value_checking(conf):
    """
    Checks the validity of the current config file.
    :param conf: the pyhocon object
    :return: True if values are correct, exits before return if any config is not correct.
    """
    # First checking the consistency of Env config.
    if conf.Env.capacity <= 0:
        sys.exit('The capacity should be larger than 0.')
    if conf.Env.nr_categories <= 0:
        sys.exit('The number of categories should be larger than 0.')
    if conf.Env.nr_categories != len(conf.Env.specific_masses):
        sys.exit("The length of specific mass array and the nr_categories do not match.")
    if conf.Env.nr_categories != len(conf.Env.specific_values):
        sys.exit("The length of specific values array and the nr_categories do not match.")
    if conf.Env.nr_categories != len(conf.Env.min_availability):
        sys.exit("The length of min_availability array and the nr_categories do not match.")
    if conf.Env.nr_categories != len(conf.Env.max_availability):
        sys.exit("The length of max_availability array and the nr_categories do not match.")
    return True
