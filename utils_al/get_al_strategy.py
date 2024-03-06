from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    KMeansSampling, KCenterGreedy, BadgeSampling, \
    NovelSamplingRandom, NovelEntropySampling, NovelMarginSampling, \
    NovelMarginSamplingAdaptive


def get_strategy(name):

    # baseline: uncertainty
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling

    # baseline: diversity
    elif name == 'KMeansSampling':
        return KMeansSampling
    elif name == 'KCenterGreedy':
        return KCenterGreedy
    elif name == 'BadgeSampling':
        return BadgeSampling

    # novel
    elif name == 'NovelSamplingRandom':
        return NovelSamplingRandom
    elif name == 'NovelEntropySampling':
        return NovelEntropySampling
    elif name == 'NovelMarginSampling':
        return NovelMarginSampling

    # adaptive (ours)
    elif name == 'NovelMarginSamplingAdaptive':
        return NovelMarginSamplingAdaptive
    else:
        raise NotImplementedError

