from path_plan_env import StaticPathPlanning, NormalizedActionsWrapper
env = NormalizedActionsWrapper(StaticPathPlanning())

env.show_map()