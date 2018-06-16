import json

import numpy as np
from keras import Model
from keras.models import load_model

from common.TokenizerSerializer import TokenizerSerializer
from common.attention import AttentionWithContext
from news_aggregator.bi_gru_classificator_baseline import get_features, get_labels, LABEL_DICT, LABEL_DICT_FULL, \
    LABEL_DICT_REV
from news_aggregator.predict import load_data


def run(tokenizer_path, weights_path):
    tokenizer = TokenizerSerializer.load(tokenizer_path)

    with open(weights_path) as f:
        model = load_model(weights_path, custom_objects={'AttentionWithContext': AttentionWithContext})

    model.summary()

    test = load_data('./data/test.csv').sample(20)
    print('Loaded {} for testing'.format(test.shape))

    test_x = get_features(test, tokenizer)
    test_y = get_labels(test).tolist()
    preds = model.predict(test_x)

    preds = np.argmax(preds, axis=1)
    y_preds = [LABEL_DICT_FULL[LABEL_DICT_REV[x]] for x in preds]
    att_layer_mod = AttentionWithContext(return_att=True)
    att_layer_mod.set_weights(model.layers[3].weights)
    att_layer_mod(model.layers[2].output)
    att_layer = Model(inputs=model.input,
                      outputs=att_layer_mod.output)
    att_scores, att_cv = att_layer.predict(test_x)
    att_scores = np.reshape(att_scores, [-1, 16])
    heatmap = []
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    reverse_word_map[0] = '<PAD>'
    for i, att_score in enumerate(att_scores):
        input_text = [reverse_word_map[s] for s in test_x[i]]

        heatmap.append({
            'text': input_text,
            'att_score': att_score.tolist(),
            'label': LABEL_DICT_FULL[test_y[i]],
            'pred': y_preds[i]
        })
        with open('attention_heatmap.json', 'w') as f:
            json.dump(heatmap, f, indent=4)
    print(heatmap)


if __name__ == '__main__':
    run('./checkpoints/2018-06-15_00:21:17/tokenizer', './checkpoints/2018-06-15_00:21:17/weights.14-0.16.hdf5')
