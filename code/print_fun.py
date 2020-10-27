# Losses that will be used to train the model
def getLossConfig():
    loss_config = {
        'cmd': 1.,
        'cub_prm': 50.,

        'xyz_prm': 50.,
        'uv_prm': 50.,
        'sym_prm': 50.,

        'cub': 1.,
        'sym_cub': 1.,
        'sq_cub': 1.,

        'leaf': 1.,
        'bb': 50.,

        'axis': 1.,
        'face': 1.,
        'align': 1.
    }

    return loss_config

def print_eval_results(eval_results, name):
        if eval_results['nc'] > 0:
            eval_results['cub_prm'] /= eval_results['nc']

        if eval_results['na'] > 0:
            eval_results['xyz_prm'] /= eval_results['na']
            eval_results['cubc'] /= eval_results['na']

        if eval_results['count'] > 0:
            eval_results['bb'] /= eval_results['count']

        if eval_results['nl'] > 0:
            eval_results['cmdc'] /= eval_results['nl']

        if eval_results['ns'] > 0:
            eval_results['sym_cubc'] /= eval_results['ns']
            eval_results['axisc'] /= eval_results['ns']

        if eval_results['np'] > 0:
            eval_results['corr_line_num'] /= eval_results['np']
            eval_results['bad_leaf'] /= eval_results['np']

        if eval_results['nsq'] > 0:
            eval_results['uv_prm'] /= eval_results['nsq']
            eval_results['sq_cubc'] /= eval_results['nsq']
            eval_results['facec'] /= eval_results['nsq']

        if eval_results['nap'] > 0:
            eval_results['palignc'] /= eval_results['nap']

        if eval_results['nan'] > 0:
            eval_results['nalignc'] /= eval_results['nan']

        eval_results.pop('nc')
        eval_results.pop('nan')
        eval_results.pop('nap')
        eval_results.pop('na')
        eval_results.pop('ns')
        eval_results.pop('nsq')
        eval_results.pop('nl')
        eval_results.pop('count')
        eval_results.pop('np')
        eval_results.pop('cub')
        eval_results.pop('sym_cub')
        eval_results.pop('axis')
        eval_results.pop('cmd')
        eval_results.pop('miss_hier_prog')

        print(
f"""

Evaluation on {name} set:

Eval {name} F-score = {eval_results['fscores']}
Eval {name} IoU = {eval_results['iou_shape']}
Eval {name} PD = {eval_results['param_dist_parts']}
Eval {name} Prog Creation Perc: {eval_results['prog_creation_perc']}
Eval {name} Cub Prm Loss = {eval_results['cub_prm']}
Eval {name} XYZ Prm Loss = {eval_results['xyz_prm']}
Eval {name} UV Prm Loss = {eval_results['uv_prm']}
Eval {name} Sym Prm Loss = {eval_results['sym_prm']}
Eval {name} BBox Loss = {eval_results['bb']}
Eval {name} Cmd Corr % {eval_results['cmdc']}
Eval {name} Cub Corr % {eval_results['cubc']}
Eval {name} Squeeze Cub Corr % {eval_results['sq_cubc']}
Eval {name} Face Corr % {eval_results['facec']}
Eval {name} Pos Align Corr % {eval_results['palignc']}
Eval {name} Neg Align Corr % {eval_results['nalignc']}
Eval {name} Sym Cub Corr % {eval_results['sym_cubc']}
Eval {name} Sym Axis Corr % {eval_results['axisc']}
Eval {name} Corr Line # % {eval_results['corr_line_num']}
Eval {name} Bad Leaf % {eval_results['bad_leaf']}""")

def print_train_results(ep_result):
    arl = 0.
    loss_config = getLossConfig()

    for loss in loss_config:
        ep_result[loss] /= bc
        if loss == 'kl':
            continue
        if torch.is_tensor(ep_result[loss]):
            arl += ep_result[loss].detach().item()
        else:
            arl += ep_result[loss]

    ep_result['recon'] = arl
    if ep_result['nl'] > 0:
        ep_result['cmdc'] /= ep_result['nl']
    if ep_result['na'] > 0:
        ep_result['cubc'] /= ep_result['na']
    if ep_result['nc'] > 0:
        ep_result['cleaf'] /= ep_result['nc']
    if ep_result['nap'] > 0:
        ep_result['palignc'] /= ep_result['nap']
    if ep_result['nan'] > 0:
        ep_result['nalignc'] /= ep_result['nan']
    if ep_result['ns'] > 0:
        ep_result['sym_cubc'] /= ep_result['ns']
        ep_result['axisc'] /= ep_result['ns']
    if ep_result['nsq'] > 0:
        ep_result['sq_cubc'] /= ep_result['nsq']
        ep_result['facec'] /= ep_result['nsq']

    ep_result.pop('na')
    ep_result.pop('nl')
    ep_result.pop('nc')
    ep_result.pop('nap')
    ep_result.pop('nan')
    ep_result.pop('np')
    ep_result.pop('ns')
    ep_result.pop('nsq')

    print(
        f"""
Recon Loss = {ep_result['recon']}
Cmd Loss = {ep_result['cmd']}
Cub Prm Loss = {ep_result['cub_prm']}
XYZ Prm Loss = {ep_result['xyz_prm']}
UV Prm Loss = {ep_result['uv_prm']}
Sym Prm Loss = {ep_result['sym_prm']}
Cub Loss = {ep_result['cub']}
Squeeze Cub Loss = {ep_result['sq_cub']}
Sym Cub Loss = {ep_result['sym_cub']}
Sym Axis Loss = {ep_result['axis']}
Face Loss = {ep_result['face']}
Leaf Loss = {ep_result['leaf']}
Align Loss = {ep_result['align']}
KL Loss = {ep_result['kl'] if 'kl' in ep_result else None}
BBox Loss = {ep_result['bb']}
Cmd Corr % {ep_result['cmdc']}
Cub Corr % {ep_result['cubc']}
Sq Cubb Corr % {ep_result['sq_cubc']}
Face Corr % {ep_result['facec']}
Leaf Corr % {ep_result['cleaf']}
Align Pos Corr = {ep_result['palignc']}
Align Neg Corr = {ep_result['nalignc']}
Sym Cub Corr % {ep_result['sym_cubc']}
Sym Axis Corr % {ep_result['axisc']}""")
