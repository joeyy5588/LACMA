from vocab import Vocab
import torch

vocab_fn = 'data/lmdb_human/data.vocab'

vocab_data = torch.load(vocab_fn)

num_to_subpolicy = {
    0: '<<pad>>',
    1: '<<seg>>',
    2: '<<goal>>',
    3: '<<mask>>',
    4: 'move forward',
    5: 'turn left',
    6: 'turn right',
    7: 'turn around',
    # 4: 'side step',
    8: 'step left',
    9: 'step right',
    10: 'step back',
    11: 'face left',
    12: 'face right',
    13: 'look up',
    14: 'look down',
    15: 'interaction',
    16: 'stop'
}
meta_action_words = list(num_to_subpolicy.values())

meta_action_vocab = Vocab(meta_action_words)

vocab_data['meta_action_vocab'] = meta_action_vocab

torch.save(vocab_data, vocab_fn)