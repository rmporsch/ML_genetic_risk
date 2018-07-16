"""Testing pytorch."""
from wepredict.wepredict import wepredict
import numpy as np
import pickle
from matplotlib import pyplot as plt


def save_pickle(object, path):
    """Save obejct."""
    pickle.dump(object, open(path, 'wb'))


def load_pickle(path):
    """Save obejct."""
    return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    testfile = 'data/1kg_LD_blocks/22_LD_block_0.npz'
    # testfile = 'data/10_LD_block_20.npz'
    # monster = wepredict(testfile, False)
    # index_valid_test = np.random.randint(0, 1092, 100)
    # alphas = np.arange(0.01, 0.2, 0.02)
    # X = monster.reader_binary(testfile)
    # y = monster.sim(X)
    # sample = monster.get_training_valid_sample(X, y, index_valid_test)
    # # model_output_l1 = compute_pytorch_l1(sample[0], sample[1], sample[2],
    # #                                   sample[3], alphas, mini_batch_size=400)
    # # model_output_l2 = compute_pytorch_l2(sample[0], sample[1], sample[2],
    # #                                    sample[3], alphas, mini_batch_size=400)
    # model_output_l0 = monster.compute_pytorch(sample['training_x'],
    #                                           sample['training_y'],
    #                                           sample['valid_x'],
    #                                           sample['valid_y'], alphas, 'l0',
    #                                           mini_batch=250, l_rate=0.001,
    #                                           epochs=201)
    # print('L1', model_output_l1['accu'])
    # print('L2', model_output_l2['accu'])
    # print('L0', model_output_l0['accu'])
    # # plot Sigmoid
    # save_pickle(model_output_l0, 'testing.pickle')
    model_output_l0 = load_pickle('testing.pickle')
    x = [x for x in range(model_output_l0['model'][0]['param']['epoch'])]
    mm = model_output_l0['model']
    # [print(k) for k in model_output_l0['accu']]
    # fig, ax = plt.subplots(len(mm), 1)
    # # ax.plot(x, model_output_l1['param']['loss'], 'r')
    # # ax.plot(x, model_output_l2['param']['loss'], 'b')
    # for u, i in enumerate(mm):
    #     ax[u].plot(x, i['param']['loss'])
    # fig.show()
    plt.plot(x, mm[0]['param']['loss'])
    plt.show()
