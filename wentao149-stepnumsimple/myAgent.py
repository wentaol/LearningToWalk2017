def get_agent_and_env(nb_steps_warmup=100, memory_limit=20000, 
                      ou_theta=0.3, ou_mu=0., ou_sigma=0.4, 
                      gamma=0.97, actor_lr=0.0003, critic_lr=0.001,
                      target_model_update=1e-3, delta_clip=1.,batch_size=32,
                      step_size=0.01, test_mode=False,
                      visualize=False,
                      memory_path="",
                      print_summary=True):
    from myProcessor import StateProcessor
    from myEnv import OsimEnv
    from myModel import getShallowModel 
    from wentaoMemory import WeaklyOrderedMemory
    from rl.random import OrnsteinUhlenbeckProcess
    from wentaoParallelDDPGAgent import ParallelDDPGAgent
    from keras.optimizers import Adam
    #if test_mode:
    #    step_size = 0.01    
    processor = StateProcessor(step_size=step_size, test_mode=test_mode)
    timestep_limit = int(1000*0.01/step_size)
    env = OsimEnv(visualize=visualize, processor=processor, step_size=step_size, timestep_limit=timestep_limit, test=test_mode)

    nb_actions = env.get_action_dim()
    actor_model, critic_model, action_input = getShallowModel(env)

    if print_summary:
        print(actor_model.summary())
        print(critic_model.summary())

    memory = WeaklyOrderedMemory(limit=memory_limit)
    print("Memory path: " + memory_path)
    if memory_path != "":
        memory.load(memory_path)
    print("Memory len: " + str(len(memory)))
    random_process = OrnsteinUhlenbeckProcess(theta=ou_theta, mu=ou_mu, sigma=ou_sigma, size=nb_actions)
    agent = ParallelDDPGAgent(nb_actions=nb_actions, actor=actor_model, critic=critic_model, 
                        critic_action_input=action_input, memory=memory, 
                        nb_steps_warmup_critic=nb_steps_warmup, nb_steps_warmup_actor=nb_steps_warmup,
                        random_process=random_process, gamma=gamma, target_model_update=target_model_update,
                        delta_clip=delta_clip, batch_size=batch_size, processor=processor)
    agent.compile([Adam(lr=actor_lr), Adam(lr=critic_lr)], metrics=['mae'])

    return (agent, env)

f = get_agent_and_env
