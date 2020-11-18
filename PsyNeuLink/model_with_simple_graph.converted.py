import psyneulink as pnl

comp = pnl.Composition(name='comp')

A = pnl.TransferMechanism(name='A', function=pnl.Linear(intercept=2.0, slope=5.0, default_variable=[[0]]), initial_value=[[0]], termination_measure=pnl.Distance(metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]))
B = pnl.TransferMechanism(name='B', function=pnl.Logistic(default_variable=[[0]]), initial_value=[[0]], termination_measure=pnl.Distance(metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]))
C = pnl.TransferMechanism(name='C', function=pnl.Exponential(default_variable=[[0]]), initial_value=[[0]], termination_measure=pnl.Distance(metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]))

comp.add_node(A)
comp.add_node(B)
comp.add_node(C)

comp.add_projection(projection=pnl.MappingProjection(name='MappingProjection from A[RESULT] to B[InputPort-0]', function=pnl.LinearMatrix(matrix=[[1.0]], default_variable=[2.0])), sender=A, receiver=B)
comp.add_projection(projection=pnl.MappingProjection(name='MappingProjection from A[RESULT] to C[InputPort-0]', function=pnl.LinearMatrix(matrix=[[1.0]], default_variable=[2.0])), sender=A, receiver=C)

comp.scheduler.add_condition(A, pnl.Always())
comp.scheduler.add_condition(B, pnl.EveryNCalls(A, 1))
comp.scheduler.add_condition(C, pnl.EveryNCalls(A, 1))

comp.scheduler.termination_conds = {pnl.TimeScale.RUN: pnl.Never(), pnl.TimeScale.TRIAL: pnl.AllHaveRun()}
comp.show_graph()