import math
import os

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

# esm if
try:
    ## Verify that pytorch-geometric is correctly installed
    import torch_geometric
    import torch_sparse
    from torch_geometric.nn import MessagePassing

    # load model
    import esm
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()

except ImportError:
    pass

# pyrsetta energy
try:
    import pyrosetta
    from pyrosetta.teaching import *
    pyrosetta.init()

except ImportError:
    pass

# proteinMPNN
try:
    from mpnn_extra import *

    # clone github repo
    sys.path.append('/home/mchungy1/scr4_jgray21/mchungy1/AbDesign/models/ProteinMPNN')
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

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

def esmif_score(pdb):

    # load pdb
    pdb_path = pdb
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    # load chains
    fpath = pdb_path
    chain_ids = ['H', 'L']
    structure = esm.inverse_folding.util.load_structure(fpath, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    # conditional sequence log-likelihoods for given BB coordinates
    target_chain_id = 'H'
    target_seq = native_seqs[target_chain_id]
    ll_h, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(
        model, alphabet, coords, target_chain_id, target_seq, padding_length=10)

    target_chain_id = 'L'
    target_seq = native_seqs[target_chain_id]
    ll_l, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(
        model, alphabet, coords, target_chain_id, target_seq, padding_length=10)

    ll_avg =(ll_l + ll_h) / 2

    perplexity = math.exp(-ll_avg)
    
    return perplexity

def pyrosetta_score(pdb):

    # load pdb into pose
    pose = pyrosetta.pose_from_pdb(pdb)

    sfxn = get_score_function(True)

    return(sfxn(pose))

def mpnn_score(pdb):

    pdb_path = pdb

    # SETUP MODEL

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_name = "v_48_020" #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

    # standard deviation of Gaussian noise to add to backbone atoms
    backbone_noise=0.00

    path_to_model_weights='/home/mchungy1/scr4_jgray21/mchungy1/AbDesign/models/ProteinMPNN/vanilla_model_weights'
    hidden_dim = 128
    num_layers = 3 
    model_folder_path = path_to_model_weights
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device) 

    noise_level_print = checkpoint['noise_level']

    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    homomer = False #@param {type:"boolean"}
    designed_chain = "H L" #@param {type:"string"}
    fixed_chain = "" #@param {type:"string"}

    if designed_chain == "":
        designed_chain_list = []
    else:
        designed_chain_list = re.sub("[^A-Za-z]+",",", designed_chain).split(",")

    if fixed_chain == "":
        fixed_chain_list = []
    else:
        fixed_chain_list = re.sub("[^A-Za-z]+",",", fixed_chain).split(",")

    chain_list = list(set(designed_chain_list + fixed_chain_list))

    num_seqs = 1 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
    num_seq_per_target = num_seqs

    #@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.
    sampling_temp = "0.1" #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]

    save_score=0                      # 0 for False, 1 for True; save score=-log_prob to npy files
    save_probs=0                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
    score_only=0                      # 0 for False, 1 for True; score input backbone-sequence pairs
    conditional_probs_only=0          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
    conditional_probs_only_backbone=0 # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)
        
    batch_size=1                      # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
    max_length=20000                  # Max sequence length
        
    out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
    jsonl_path=''                     # Path to a folder with parsed pdb into jsonl
    omit_AAs='X'                      # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.
        
    pssm_multi=0.0                    # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
    pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs
    pssm_log_odds_flag=0               # 0 for False, 1 for True
    pssm_bias_flag=0                   # 0 for False, 1 for True

    folder_for_outputs = out_folder

    NUM_BATCHES = num_seq_per_target//batch_size
    BATCH_COPIES = batch_size
    temperatures = [float(item) for item in sampling_temp.split()]
    omit_AAs_list = omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    chain_id_dict = None
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None
    bias_AA_dict = None
    tied_positions_dict = None
    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))

    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    for chain in chain_list:
        l = len(pdb_dict_list[0][f"seq_chain_{chain}"])

    if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
    else:
        tied_positions_dict = None

    # RUN
    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
            name_ = batch_clones[0]['name']

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()

            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    if tied_positions_dict == None:
                        sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all)
                        S_sample = sample_dict["S"] 
                    else:
                        sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all)
                    # Compute scores
                        S_sample = sample_dict["S"]
                    log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    all_log_probs_list.append(log_probs.cpu().data.numpy())
                    S_sample_list.append(S_sample.cpu().data.numpy())
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        score_list.append(score)
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j==0 and temp==temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                l0 += 1
                            sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                            print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                            sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                            print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                            native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + '/' + seq[l0:]
                            l0 += 1
                        score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                        seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
    
    perplexity = math.exp(score)

    return perplexity
