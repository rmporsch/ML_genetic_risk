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
    monster = wepredict(testfile, False)
    train_index, valid_index, test_index = monster.generate_valid_test_data(
        1092, 0.20, 0.10)
    alphas = np.arange(0.01, 0.2, 0.02)
    X = monster.reader_binary(testfile)
    y = monster.sim(X)
    sample = monster.get_training_valid_test_sample(X, y, train_index,
                                                    valid_index,
                                                    test_index)
    model_output_l1 = monster.compute_pytorch(sample['training_x'],
                                              sample['training_y'],
                                              sample['valid_x'],
                                              sample['valid_y'], alphas, 'l1',
                                              mini_batch=500, l_rate=0.001,
                                              epochs=301)
    model_output_l0 = monster.compute_pytorch(sample['training_x'],
                                              sample['training_y'],
                                              sample['valid_x'],
                                              sample['valid_y'], alphas, 'l0',
                                              mini_batch=500, l_rate=0.001,
                                              epochs=301)
    print('L1', model_output_l1['accu'])
    # print('L2', model_output_l2['accu'])
    print('L0', model_output_l0['accu'])
    # plot Sigmoid
    save_pickle(model_output_l0, 'testing_l0.pickle')
    save_pickle(model_output_l1, 'testing_l1.pickle')
    # model_output_l0 = load_pickle('testing_l0.pickle')
    # model_output_l1 = load_pickle('testing_l1.pickle')
    best_l0 = np.argmax(model_output_l0['accu'])
    best_l1 = np.argmax(model_output_l1['accu'])
    print('test L0', model_output_l0['accu'][best_l0])
    print('test L1', model_output_l1['accu'][best_l1])
    out_l0 = monster.evaluate_test(X[test_index, :], y[test_index],
                                model_output_l0['model'][best_l0]['coef'][1])
    out_l1 = monster.evaluate_test(X[test_index, :], y[test_index],
                                model_output_l1['model'][best_l1]['coef'][0])
    print('L1', out_l1['accu'])
    print('L0', out_l0['accu'])
    x = [x for x in range(model_output_l0['model'][0]['param']['epoch'])]
    fig, ax = plt.subplots(1, 2)
    best = [best_l0, best_l1]
    for i, m in enumerate([model_output_l0, model_output_l1]):
        mm = m['model']
        ax[i].plot(x, mm[best[i]]['param']['loss'])
    fig.show()
