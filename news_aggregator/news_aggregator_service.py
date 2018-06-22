from news_aggregator.news_aggregator_predictor import CategoryPredictor

category_TOKENIZER = '../../news_aggregator/checkpoints/2018-06-16_09:25:03/tokenizer'
category_MODEL = '../../news_aggregator/checkpoints/2018-06-16_09:25:03/weights.11-0.16.hdf5'

category_clf = CategoryPredictor(tokenizer_path=category_TOKENIZER,
                                 weights_path=category_MODEL)
