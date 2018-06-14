import json

import numpy as np
from keras import Model
from keras.models import load_model

from common.TokenizerSerializer import TokenizerSerializer
from common.attention import AttentionWithContext
from news_aggregator.bi_gru_classificator_baseline import get_features, get_labels, LABEL_DICT, LABEL_DICT_FULL
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

    LABEL_DICT_REV = {val: key for key, val in LABEL_DICT.items()}
    preds = np.argmax(preds, axis=1)
    y_preds = [LABEL_DICT_FULL[LABEL_DICT_REV[x]] for x in preds]
    att_layer = Model(inputs=model.input,
                      outputs=model.layers[4].output)
    att_scores = att_layer.predict(test_x)
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
    run('./checkpoints/2018-05-20_14:35:26/tokenizer', './checkpoints/2018-05-20_14:35:26/weights.14-0.16.hdf5')
