#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
import tqdm
from fairseq.data.encoders.gpt2_bpe import get_encoder
from threading import Semaphore
import glob

def yield_from_files(paths):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.
    :param fnames: list of filenames
    """
    def yielder(fname):
        # open the file and then call .read() to get the text 
        with open(fname) as f:
            yield f.readlines()
        # for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
        #     semaphore.acquire()
        #     # print(len(f))
        #     yield f
    
    for path in paths:
        files = glob.glob(path + "/*")
        # iterate over the list getting each file 
        for fname in files:
            # semaphore.acquire()
            yield from yielder(fname)
# def yield_from_files(fnames: list, semaphore):
#     """
#     Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
#     other compressed formats. Also filters out empty documents.
#     :param fnames: list of filenames
#     """
    

#     def yielder(fname, semaphore):
#         # open the file and then call .read() to get the text 
#         with open(fname) as f:
#             for line in f:
#                 yield line
#         # for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
#         #     semaphore.acquire()
#         #     # print(len(f))
#         #     yield f
    
#     files = glob.glob(path)
#     # iterate over the list getting each file 
#     for fle in files:
#     for fname in fnames:
#         semaphore.acquire()
#         yield from yielder(fname, semaphore)

def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=100)
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        

        # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
        # hence building up memory
        # semaphore = Semaphore(10000 + args.workers)
        f_lines = yield_from_files(args.inputs)
        # for i, f_line in enumerate(f_lines):
        #     print(f_line)
        #     if i > 5:
        #         exit(0)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, f_lines, 25)

        stats = Counter()
        # print(len(encoded_lines))
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            # print(enc_lines)
            # release semaphore so `yield_from_files` can add another file to the buffer
            # semaphore.release()
            if filt == "PASS":
                for enc_line in enc_lines:
                    print(enc_line, file=outputs[0])
            else:
                stats["num_filtered_" + filt] += 1
            if i % 500 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            # if len(line) == 0 and not self.args.keep_empty:
            #     return ["EMPTY", None]
            if len(line) == 0:
                tokens = self.encode('')
                enc_lines.append(" ".join(tokens))
            else:
                tokens = self.encode(line)
                enc_lines.append(" ".join(tokens))
        # tokens = self.encode('')
        # enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
