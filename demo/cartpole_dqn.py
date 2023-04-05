from tqdm import trange

from matex.agents import DQN


def train():
    import gymnasium as gym
    import torch
    from torch.utils.tensorboard import SummaryWriter

    env = gym.make("CartPole-v1", render_mode=None)
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = 1e-4
    GAMMA = 0.99
    MEMORY_SIZE = 10000
    BATCH_SIZE = 128
    N_EPISODES = 10
    MAX_STEPS = 200
    EPSILON = 0.1
    EPSILON_DECAY = 0.99
    EPSILON_MIN = 0.01
    HIDDEN_SIZE = 64
    TARGET_NET_UPDATE_FREQ = 10
    TAU = 0.01
    best_step = 0

    agent = DQN(
        lr=LR,
        gamma=GAMMA,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=HIDDEN_SIZE,
        device=device,
    )

    with trange(N_EPISODES) as pbar:
        for ep in pbar:
            pbar.set_description(f"Episode: {ep:>5}")

            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)

            for step in range(MAX_STEPS):
                action = agent.act(
                    state,
                    eps=EPSILON,
                    prog_rate=ep / N_EPISODES,
                    eps_decay=EPSILON_DECAY,
                    eps_min=EPSILON_MIN,
                )
                next_state, _, done, _, _ = env.step(action.item())
                next_state = torch.tensor(next_state, device=device, dtype=torch.float).view(1, -1)

                if done:
                    if step >= MAX_STEPS - 5:
                        reward = torch.tensor([[1.0]], device=device, dtype=torch.float).view(1, 1)
                    else:
                        reward = torch.tensor([[-1.0]], device=device, dtype=torch.float).view(1, 1)
                else:
                    reward = torch.tensor([[0.01]], device=device, dtype=torch.float).view(1, 1)

                agent.memorize(state, action, next_state, reward, done)
                agent.learn()

                state = next_state

                if step % TARGET_NET_UPDATE_FREQ == 0:
                    agent.update_target_network(soft_update=True, tau=TAU)

                if done:
                    writer.add_scalar("step", step + 1, ep)
                    if step + 1 >= best_step:
                        best_step = step + 1
                        agent.save("./runs/dqn.pth")
                    break

            pbar.set_postfix(step=f"{step+1:*>3}")

    writer.close()

    def play():
        import gym
        import moviepy.editor as mpy

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, "./runs/")
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)
        done = False

        while not done:
            action = agent.act(state, deterministic=True)
            state, _, done, _, _ = env.step(action.item())
            state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)

        movie = mpy.VideoFileClip("./runs/rl-video-episode-0.mp4")
        movie.write_gif("./runs/rl-video-episode-0.gif")

    play()


if __name__ == "__main__":
    train()
