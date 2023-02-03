from typing import Dict
import argparse

# from https://stackoverflow.com/a/49753634
def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            if group_action.dest == arg:
                action._group_actions.remove(group_action)
                return

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

# a wrapper of add_argument() method that handles dest variable automatically.
def add_nested_argument(parser, name, struct_name="attn_args", prefix="", **kwargs):
    if len(prefix) == 0:
        dest_name = '{}.{}'.format(struct_name, name.lstrip('-').replace('-', '_'))
    else:
        dest_name = '{}.{}'.format(struct_name, remove_prefix(name, "--" + prefix + "-").replace('-', '_'))
    parser.add_argument(name, dest=dest_name, **kwargs)

# copied from https://stackoverflow.com/a/18709860;
# a useful variant that supports nested arg parsing!
class NestedNamespace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, NestedNamespace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value



from .abstract_attention import MultiheadAttention
from .local_attention import LocalAttention
from .kernelized_attention import KernelizedAttention
from .lara import LinearRA
from .randomized_attention import RandomizedAttention
from .scatterbrain_attention import ScatterBrain
from .eva import EVA
from .causal_eva import CausalEVAttention

class AttentionFactory(object):
    attn_dict = {
        'performer': KernelizedAttention,
        'softmax': MultiheadAttention,
        'local': LocalAttention,
        'lara': LinearRA,
        'ra': RandomizedAttention,
        'scatterbrain': ScatterBrain,
        'eva': EVA,
        'causal_eva': CausalEVAttention,
    }
    def __init__(self):
        super(AttentionFactory, self).__init__()

    @classmethod
    def build_attention(
        cls,
        attn_name: str,
        attn_args: Dict):
        attn_cls = cls.attn_dict[attn_name]
        return attn_cls(**attn_args)

    @classmethod
    def add_attn_specific_args(cls, parent_parser, attn_name, struct_name="attn_args", prefix=""):
        if hasattr(cls.attn_dict[attn_name], 'add_attn_specific_args'):
            return cls.attn_dict[attn_name].add_attn_specific_args(parent_parser, struct_name=struct_name, prefix=prefix)
        else:
            return parent_parser

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     add_nested_argument(parser, "--decoding-window-size", "attn_args_decoder", "decoding", default=15, type=int)
#     add_nested_argument(parser, "--encoding-window-size", "attn_args_encoder", "encoding", default=15, type=int)
#     args = parser.parse_args(namespace=NestedNamespace())
#     print(vars(args))