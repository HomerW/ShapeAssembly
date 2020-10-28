from random import random, shuffle
import torch
from ShapeAssembly import Program, ShapeAssembly, make_hier_prog, hier_execute
from valid import check_stability, check_rooted
from utils import writeHierProg

def writeSPC(pc, fn):
    with open(fn, 'w') as f:
        for a,b,c in pc:
            f.write(f'v {a} {b} {c} \n')

def writeObj(verts, faces, outfile):
    faces = faces.clone()
    faces += 1
    with open(outfile, 'w') as f:
        for a, b, c in verts.tolist():
            f.write(f'v {a} {b} {c}\n')

        for a, b, c in faces.tolist():
            f.write(f"f {a} {b} {c}\n")

def samp(n, m, s):
    return (((torch.rand(n) - .5) * 2 * s) + m).tolist()

def noise(n):
    return (torch.randn(1) * n).item()

def clamp(n, _min, _max):
    return min(max(n, _min), _max)

def make_cuboid(n, dims, d):
    a = round(clamp(dims[0], 0.05, 3.0), 3)
    b = round(clamp(dims[1], 0.05, 3.0), 3)
    c = round(clamp(dims[2], 0.05, 3.0), 3)
    return f"{n} = Cuboid({a}, {b}, {c}, {d})"

def make_attach(att):
    for i in range(2, 8):
        att[i] = round(clamp(att[i], 0.0, 1.0), 3)
    a, b, x, y, z, u, v, w = att
    return f"attach({a}, {b}, {x}, {y}, {z}, {u}, {v}, {w})"

def make_squeeze(sq):
    for i in (4, 5):
        sq[i] = round(clamp(sq[i], 0.0, 1.0), 3)
    return f"squeeze({sq[0]}, {sq[1]}, {sq[2]}, {sq[3]}, {sq[4]}, {sq[5]})"

def nfilter(list):
    return [l for l in list if l is not None]

def make_body():
    base_dims = [s + noise(.2) for s in samp(3, 0.5, 0.2)]
    base_dims[0] += .3
    base_dims[2] += .2

    if random() >= .2:
        mid_dims = [s + noise(.2) for s in samp(3, 0.5, 0.2)]

        if random() >= 0.5:
            mid_dims[0] = base_dims[0] + noise(.05)

        if random() >= 0.5:
            mid_dims[2] = base_dims[2] + noise(0.05)

        use_mid = True
    else:
        mid_dims = [0, 0, 0]
        use_mid = False

    top_dims = [s + noise(.2) for s in samp(3, 0.5, 0.2)]

    if use_mid and random() >= 0.5:
        top_dims[0] = mid_dims[0] + noise(.05)

    if use_mid and random() >= 0.5:
        top_dims[2] = mid_dims[2] + noise(.05)

    bbox_dims = [
        max(base_dims[0], mid_dims[0], top_dims[0]) + noise(.1),
        base_dims[1] + mid_dims[1] + top_dims[1] + noise(.1),
        max(base_dims[2], mid_dims[2], top_dims[2] + noise(.1))
    ]

    at1 = [
        'base',
        'bbox',
        0.5 + noise(0.05),
        0.0 + noise(0.05),
        0.5 + noise(0.05),
        0.5 + noise(.1),
        0.0 + noise(0.05),
        0.5 + noise(.1)
    ]

    at2 = [
        'top',
        'bbox',
        0.5 + noise(0.05),
        1.0 + noise(0.05),
        0.5 + noise(0.05),
        0.5 + noise(.1),
        1.0 + noise(0.05),
        random()
    ]

    if use_mid:
        at3 = [
            'mid',
            'base',
            0.5 + noise(.05),
            0.0 + noise(.05),
            0.5 + noise(.05),
            0.5 + noise(.2),
            1.0 + noise(.05),
            0.5 + noise(.2)
        ]

        if random() < 0.5:
            at4 = [
                'top',
                'mid',
                0.5  + noise(.05),
                0.0  + noise(.05),
                0.5  + noise(.05),
                0.5 + noise(.2),
                1.0 + noise(.05),
                0.5 + noise(.2)
            ]

        else:
            at4 = [
                'mid',
                'top',
                0.5  + noise(.05),
                1.0  + noise(.05),
                0.5  + noise(.05),
                0.5 + noise(.2),
                0.0 + noise(.05),
                0.5 + noise(.2)
            ]


    else:
        if random() < 0.5:
            at3 = [
                'top',
                'base',
                0.5 + noise(.05),
                0.0 + noise(.05),
                0.5 + noise(.05),
                noise(.2),
                1.0 + noise(.05),
                noise(.2)
            ]
        else:
            at3 = [
                'base',
                'top',
                0.5 + noise(.05),
                1.0 + noise(.05),
                0.5 + noise(.05),
                0.5 + noise(.2),
                0.0 + noise(.05),
                0.5 + noise(.2)
            ]

        at4 = None

    clines = nfilter([
        make_cuboid('bbox', bbox_dims, True),
        make_cuboid('base', base_dims, random() >= 0.2),
        make_cuboid('top', top_dims, random() >= 0.5),
        make_cuboid('mid', mid_dims, random() >= 0.5) if use_mid else None
    ])

    alines = nfilter([
        make_attach(at1),
        make_attach(at2),
        make_attach(at3),
        make_attach(at4) if at4 is not None else None
    ])

    return clines, alines, use_mid

def add_extensor(par, name):
    dims = [s + noise(.1) for s in samp(3, 0.4, 0.3)]
    clines = [make_cuboid(name, dims, True)]
    att = [
        0.5 + noise (.05),
        0.5 + noise (.05),
        0.5 + noise (.05),
        0.5 + noise (.05),
        0.5 + noise (.05),
        0.5 + noise (.05),
    ]

    if name == 'left' or name == 'right':
        rand_inds = [4,5]
        a_ind = [0,3]
        if name == 'left':
            v = [0, 1]
        else:
            v = [1, 0]

    elif name == 'top' or name == 'bot':
        rand_inds = [3,5]
        a_ind = [1,4]
        if name == 'bot':
            v = [0, 1]
        else:
            v = [1, 0]

    elif name == 'front' or name == 'back':
        rand_inds = [3,4]
        a_ind = [2,5]
        if name == 'back':
            v = [0, 1]
        else:
            v = [1, 0]

    for r in rand_inds:
        att[r] = random()

    for i, _v in zip(a_ind, v):
        att[i] = _v + noise(0.05)

    alines = [make_attach([name, par] + att)]

    return clines, alines

def make_skel():
    clines, alines, has_mid = make_body()

    slines = []

    if has_mid:
        if random() >= 0.5:
            # add a left-right extensor
            r = random()
            if r >= 0.7:
                _clines, _alines = add_extensor('mid', 'left')
                slines.append(f'reflect(left, X)')
            elif r >= 0.8:
                _clines,_alines = add_extensor('mid', 'left')
            elif r >= 0.9:
                _clines, _alines = add_extensor('mid', 'right')
            else:
                _clines, _alines = add_extensor('mid', 'left')
                __clines, __alines = add_extensor('mid', 'right')
                _clines += __clines
                _alines += __alines

            clines += _clines
            alines += _alines

        if random() >= 0.8:
            r = random()
            if r >= 0.4:
                _clines, _alines = add_extensor('mid', 'front')
            elif r >= 0.8:
                _clines, _alines = add_extensor('mid', 'back')
            elif r >= 0.9:
                _clines, _alines = add_extensor('mid', 'back')
                slines.append(f'reflect(back, Z)')
            else:
                _clines, _alines = add_extensor('mid', 'back')
                __clines, __alines = add_extensor('mid', 'front')
                _clines += __clines
                _alines += __alines

            clines += _clines
            alines += _alines

    return clines + alines + slines, has_mid


def sample_pc(params, i, j):
    scene_geom = params[:12]

    xyz = ((torch.rand(1,200, 3) * 100.).round()/100.)
    xyz[:,:,i] = j

    s_r = torch.cat(
            (
                (scene_geom[6:9] / (scene_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
                (scene_geom[9:12] / (scene_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)).unsqueeze(1),
                torch.cross(
                    scene_geom[6:9] / (scene_geom[6:9].norm(dim=0).unsqueeze(0) + 1e-8),
                    scene_geom[9:12] / (scene_geom[9:12].norm(dim=0).unsqueeze(0) + 1e-8)
                ).unsqueeze(1)
            ), dim = 1)

    pts = ((s_r @ (((xyz - .5) * scene_geom[:3]).unsqueeze(-1))).squeeze() + scene_geom[3:6]).squeeze()
    return pts

def getInside(c, r, pts):
    O = c.getPos(
        torch.tensor(0.),
        torch.tensor(0.),
        torch.tensor(0.)
    )
    A = torch.stack([
        c.dims[0] * c.rfnorm,
        c.dims[1] * c.tfnorm,
        c.dims[2] * c.ffnorm
    ]).T

    B = pts.T - O.unsqueeze(1)
    p = A.inverse() @ B
    ins = (p.T - .5).abs().max(dim=1).values <= 0.5

    O = r.getPos(
        torch.tensor(0.),
        torch.tensor(0.),
        torch.tensor(0.)
    )
    A = torch.stack([
        r.dims[0] * r.rfnorm,
        r.dims[1] * r.tfnorm,
        r.dims[2] * r.ffnorm
    ]).T

    B = pts.T - O.unsqueeze(1)
    p = A.inverse() @ B
    return p.T[ins]

def make_upd_att(a, b, c, u, v, tn=0., bn=0.):
    if random() >= 0.5:
        return [make_squeeze([a, b, c, 'top', u, v])]
    else:
        return [
            make_attach([
                a,
                b,
                0.5 + noise(.05),
                1.0 + noise(.05),
                0.5 + noise(.05),
                u + noise(tn),
                1.0 + noise(.05),
                v + noise(tn),
            ]),
            make_attach([
                a,
                c,
                0.5 + noise(.05),
                0.0 + noise(.05),
                0.5 + noise(.05),
                u + noise(bn),
                0.0 + noise(.05),
                v + noise(bn),
            ])
        ]

def make_ref_base_prog(rel_ins, base_dims):
    lines = []

    i_dims = [
        base_dims[0].item() * (.25 + min(noise(.1), .1)),
        base_dims[1].item(),
        base_dims[2].item()
    ]

    align = random() >= 0.8

    iconn_count = 0
    if random() >= 0.75:
        iconn_count += 1
        if random() >= 0.5:
            iconn_count += 1

    ic1_dims = [
        base_dims[0].item() - (i_dims[0] * 2) + noise(.05),
        base_dims[1].item() * noise(.1),
        base_dims[2].item() * noise(.1)
    ]

    ic2_dims = [
        base_dims[0].item() - (i_dims[0] * 2) + noise(.05),
        base_dims[1].item()  * noise(.1),
        base_dims[2].item() * noise(.1)
    ]

    lines += nfilter([
        make_cuboid('bbox', [b.item() for b in base_dims], True),
        make_cuboid('ileg', i_dims, align),
        make_cuboid('iconn_1', ic1_dims, True) if iconn_count > 0 else None,
        make_cuboid('iconn_2', ic2_dims, True) if iconn_count > 1 else None,
    ])

    lines += make_upd_att(
        'ileg', 'bbox', 'bbox',
        rel_ins[:,0].min().item()+i_dims[0]+noise(.05),
        rel_ins[:,2].mean().item()+noise(.05),
        0.,
        0.05
    )

    lines.append('reflect(ileg, X)')

    if iconn_count > 0:
        lines += [
            make_attach([
                'iconn_1',
                'ileg',
                0.0+noise(.05),
                0.5+noise(.05),
                0.5+noise(.05),
                1.0+noise(.05),
                clamp(random(), .1, .9),
                clamp(random(), .1, .9)
            ])]

    if iconn_count > 1:
        lines += [
            make_attach([
                'iconn_2',
                'ileg',
                0.0+noise(.05),
                0.5+noise(.05),
                0.5+noise(.05),
                1.0+noise(.05),
                clamp(random(), .1, .9),
                clamp(random(), .1, .9)
            ])]

    if random() >= 0.75:
        return ['Assembly Program_1 {'] +\
            ['\t' + b for b in lines] +\
            ['}']

    lines = [l.replace('ileg', 'Program_2') for l in lines]

    s_lines = []

    s_dims = [
        i_dims[0],
        i_dims[1],
        i_dims[2] * (.25 + min(noise(.1), .1)),
    ]

    sc_dims = [
        i_dims[0] * (noise(.1) + .5),
        i_dims[1] * (.1 + noise(.1)),
        i_dims[2] - (s_dims[2] * 2) + noise(.05),
    ]

    sconn = random() >= 0.75

    s_lines += nfilter([
        make_cuboid('bbox', [b for b in i_dims], True),
        make_cuboid('sleg', s_dims, True),
        make_cuboid('sconn', sc_dims, True) if sconn else None,
    ])
    s_lines += make_upd_att('sleg', 'bbox', 'bbox', 0.5, rel_ins[:,2].min().item(), 0., 0.0)

    if sconn:
        s_lines += [
            make_attach([
                'sconn',
                'sleg',
                0.5+noise(.05),
                0.5+noise(.05),
                0.0+noise(.05),
                clamp(random(), .1, .9),
                clamp(random(), .1, .9),
                1.0+noise(.05),
            ])]

    s_lines.append('reflect(sleg, Z)')

    return ['Assembly Program_1 {'] +\
            ['\t' + b for b in lines] +\
            ['}'] +\
            ['Assembly Program_2 {'] +\
            ['\t' + s for s in s_lines] +\
            ['}']


def make_sl_base_prog(rel_ins, base_dims):

    l_dims = [
        base_dims[0].item() * noise(.2),
        base_dims[1].item(),
        base_dims[2].item() * noise(.2),
    ]

    align = random() >= 0.5

    tleg = random() >= 0.5

    clines = nfilter([
        make_cuboid('bbox', [b.item() for b in base_dims], True),
        make_cuboid('leg1', l_dims, align),
        make_cuboid('leg2', l_dims, align),
        make_cuboid('leg3', l_dims, align),
        make_cuboid('leg4', l_dims, align) if not tleg else None,
    ])

    min_u = rel_ins[:,0].min().item()
    max_u = rel_ins[:,0].max().item()

    min_v = rel_ins[:,2].min().item()
    max_v = rel_ins[:,2].max().item()

    min_u += l_dims[0]
    min_v += l_dims[2]

    max_u -= l_dims[0]
    max_v -= l_dims[2]

    if not tleg:
        alines = make_upd_att('leg1', 'bbox', 'bbox', min_u, min_v, 0., 0.05) + \
                 make_upd_att('leg2', 'bbox', 'bbox', min_u, max_v, 0., 0.05) + \
                 make_upd_att('leg3', 'bbox', 'bbox', max_u, min_v, 0., 0.05) + \
                 make_upd_att('leg4', 'bbox', 'bbox', max_u, max_v, 0., 0.05)
    else:
        corners = [(min_u, min_v), (min_u, max_v), (max_u, min_v), (max_u, max_v)]
        shuffle(corners)
        alines = make_upd_att('leg1', 'bbox', 'bbox', corners[0][0], corners[0][1], 0., 0.05) + \
                 make_upd_att('leg2', 'bbox', 'bbox', corners[1][0], corners[1][1], 0., 0.05) + \
                 make_upd_att('leg3', 'bbox', 'bbox', (corners[2][0] + corners[3][0])/2, (corners[2][1] + corners[3][1])/2,  0., 0.05)

    lines = clines + alines
    return ['Assembly Program_1 {'] +\
        ['\t' + b for b in lines] +\
        ['}']

# Take in dims of base
# Take in base cuboid and mid/top cuboid after exec,
def make_base_prog(base, par):
    spts = sample_pc(base.getParams(), 1, 1.0)
    rel_ins = getInside(par, base, spts)
    if rel_ins.sum() == 0:
        assert False, 'no connections in base prog'

    mode = 'ref' if random() >= 0.5 else 'sing'

    if mode == 'sing':

        return make_sl_base_prog(
            rel_ins,
            base.dims
        )

    elif mode == 'ref':
        return make_ref_base_prog(
            rel_ins,
            base.dims
        )

def main(ind):
    sa = ShapeAssembly()
    root_lines, has_mid = make_skel()

    prog_lines = ['Assembly Program_0 {']
    for b in root_lines:
        prog_lines.append('\t'+b)
    prog_lines.append('}')

    RP = Program()

    for l in root_lines:
        RP.execute(l)

    base_par = 'mid' if has_mid else 'top'

    base_lines = make_base_prog(
        RP.cuboids['base'],
        RP.cuboids[base_par]
    )

    for i in range(len(prog_lines)):
        prog_lines[i] = prog_lines[i].replace('base', 'Program_1')



    prog_lines += base_lines

    #for b in base_lines:
    #    prog_lines.append('\t'+b)
    #prog_lines.append('}')

    cube_count = -1
    switches = []
    for line in prog_lines:
        if 'Cuboid' in line:
            if not ('Program_' in line or "bbox" in line):
                switches.append((
                    f'cube{cube_count}', line.split()[0]
                ))
            if "bbox" in line:
                cube_count = -1

            cube_count += 1
    for a, b in switches:
        prog_lines = [line.replace(b,a) for line in prog_lines]

    hier_prog = make_hier_prog(prog_lines)
    verts, faces = hier_execute(hier_prog)
    if check_rooted(verts, faces) and check_stability(verts, faces):
        # writeObj(verts, faces, f'out_{ind}.obj')
        writeHierProg(hier_prog, f"random_hier_data/{ind}.txt")
        return True
    return False

count = 0
while(count < 3000):
    print(count)
    try:
        r = main(count)
        if r:
            count += 1
    except Exception as e:
        print(e)
