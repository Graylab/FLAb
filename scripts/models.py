import math

# iglm
try:
    import iglm
    from iglm import IgLM
    iglm = IgLM()

except ImportError:
    pass

# antiberty
try:
    from antiberty import AntiBERTyRunner
    antiberty = AntiBERTyRunner()

except ImportError:
    pass

# progen
try:
    from progen_extra import *

except ImportError:
    pass


def iglm_score(df):
    """
    input: df with columns: heavy,
                            light,
                            fitness
    output: df with columns: heavy,
                             light,
                             fitness,
                             heavy_perplexity,
                             light_perplexity,
                             average_perplexity
    """

    # score heavy sequences
    heavy_score = []

    for seq in df['heavy']:
        sequence = seq
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"

        log_likelihood = iglm.log_likelihood(
            sequence,
            chain_token,
            species_token,
        )

        perplexity = math.exp(-log_likelihood)
        heavy_score.append(perplexity)

    df['heavy_perplexity'] = heavy_score

    # score light sequences
    light_score = []

    for seq in df['light']:
        sequence = seq
        chain_token = "[LIGHT]"
        species_token = "[HUMAN]"

        log_likelihood = iglm.log_likelihood(
            sequence,
            chain_token,
            species_token,
        )

        perplexity = math.exp(-log_likelihood)
        light_score.append(perplexity)

    df['light_perplexity'] = light_score

    df['average_perplexity'] = (df['heavy_perplexity'] + df['light_perplexity']) / 2

    return df


def antiberty_score(df):
    heavy_score = []
    light_score = []

    for row in range(len(df)):
        sequences = [
            df['heavy'][row],
            df['light'][row],
        ]

        pll = antiberty.pseudo_log_likelihood(sequences, batch_size=16)

        perplexity_h = math.exp(-pll.tolist()[0])
        perplexity_l = math.exp(-pll.tolist()[1])

        heavy_score.append(perplexity_h)
        light_score.append(perplexity_l)

    df['heavy_perplexity'] = heavy_score
    df['light_perplexity'] = light_score

    df['average_perplexity'] = (df['heavy_perplexity'] + df['light_perplexity']) / 2

    return df

def progen_score(df, model_version, device):
    ### main
    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B

    # (2) preamble
    set_env()
    set_seed(42, deterministic=True)

    ### WILL HAVE TO EDIT TO MAKE GPU A FLAG
    if torch.cuda.is_available():
        print('gpu is available')

    else:
        print('falling back to cpu')

    device = torch.device(device)
    ckpt = f"/home/mchungy1/scr16_jgray21/mchungy1/progen/progen2/checkpoints/progen2-{model_version}"

    if device.type == 'cpu':
        print('falling back to fp32')
        fp16 = False

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=True).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='/home/mchungy1/scr16_jgray21/mchungy1/progen/progen2/tokenizer.json')

    def ce(tokens):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16):
                target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                logits = model(target, labels=target).logits

                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                return cross_entropy(logits=logits, target=target).item()

    def ll(tokens, f=log_likelihood, reduction='mean'):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                logits = model(target, labels=target).logits

                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                # remove terminals
                bos_token, eos_token = 3, 4
                if target[-1] in [bos_token, eos_token]:
                    logits = logits[:-1, ...]
                    target = target[:-1]

                assert (target == bos_token).sum() == 0
                assert (target == eos_token).sum() == 0

                # remove unused logits
                first_token, last_token = 5, 29
                logits = logits[:, first_token:(last_token+1)]
                target = target - first_token

                assert logits.shape[1] == (last_token - first_token + 1)

                return f(logits=logits, target=target, reduction=reduction).item()

    # score heavy sequences using progen
    perplexity_mean_list_h = []

    for seq in df['heavy']:
        context = seq
        
        reverse = lambda s: s[::-1]

        ll_lr_mean = ll(tokens=context, reduction='mean')
        ll_rl_mean = ll(tokens=reverse(context), reduction='mean')

        ll_mean = .5 * (ll_lr_mean + ll_rl_mean)
        
        perplexity = math.exp(-ll_mean)
        
        perplexity_mean_list_h.append(perplexity)

    df['heavy_perplexity'] = perplexity_mean_list_h

    # score light sequences using progen
    perplexity_mean_list_l = []

    for seq in df['light']:
        context = seq

        reverse = lambda s: s[::-1]

        ll_lr_mean = ll(tokens=context, reduction='mean')
        ll_rl_mean = ll(tokens=reverse(context), reduction='mean')

        ll_mean = .5 * (ll_lr_mean + ll_rl_mean)
    
        perplexity = math.exp(-ll_mean)
    
        perplexity_mean_list_l.append(perplexity)

    df['light_perplexity'] = perplexity_mean_list_l

    df['average_perplexity'] = (df['heavy_perplexity'] + df['light_perplexity']) / 2

    return df

    