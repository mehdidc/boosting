from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample

max_nb_layers = 4

corruption = lambda name : hp.uniform(name, 0, 1)
learning_rate = lambda name: hp.uniform(name, 0, 1)
nb_neurons = lambda name: hp.quniform(name, 100, 1500, 10)

templates = {
        "corruption": corruption,
        "learning_rate": learning_rate,
        "nb_neurons": nb_neurons
}


args = []
prec_vars = {}
for depth in xrange(1, max_nb_layers + 1):
    vars = {}
    vars.update(prec_vars)
    prec_vars = vars
    vars["depth"] = depth
    for name, template in templates.items():
        name_cur_depth = name + "%d" % (depth,)
        vars[name_cur_depth] = template(name_cur_depth)
    args.append(vars)

def from_dict(d):
    depth = d["depth"]
    corruption = [None] * depth

    values = {}
    for name in templates.keys():
        values[name] = [None] * depth

    for name, value in d.items():
        for template_name in templates.keys():
            if name.startswith(template_name):
                layer = int(name[len(template_name):])
                values[template_name][layer - 1] = value
                
    return values
space = scope.switch(
         hp.randint('d', 3) + 1,
         *args
 )


def fn(args):
    print args
    return {"loss" : 1, "status": STATUS_OK, "cost": 150

trials = Trials()
best = fmin(fn=fn, space=space, max_evals=10, algo=tpe.suggest, trials=trials)
print trials.results
