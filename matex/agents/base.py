class Agent:
    def act(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def memorize(self):
        raise NotImplementedError

    def on_step_start(self):
        raise NotImplementedError

    def on_episode_start(self):
        raise NotImplementedError

    def on_step_end(self):
        raise NotImplementedError

    def on_episode_end(self):
        raise NotImplementedError
