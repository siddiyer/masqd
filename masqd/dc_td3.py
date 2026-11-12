import copy

import torch
import torch.nn.functional as F

from .networks import Actor, DCCritic


class DCTD3:
    """
    TD3 with a descriptor-conditioned critic. The critic is shared by all agents; each agent uses its own
    head of the team actor. The keyword-only arguments are the dc_td3 section of a config.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        num_agents,
        num_bds,
        device,
        actor_hidden_size,
        critic_hidden_size,
        *,
        max_action,
        discount,
        tau,
        policy_noise,
        noise_clip,
        policy_freq,
        lr,
        batch_size,
        similarity_scale,
        refine_steps,
        refine_lr,
        refine_batch_size,
    ):
        self.device = device
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.lr = lr
        self.batch_size = batch_size
        self.similarity_scale = similarity_scale
        self.refine_steps = refine_steps
        self.refine_lr = refine_lr
        self.refine_batch_size = refine_batch_size

        # train_critic() replaces this actor with a copy of the policy being trained.
        self.set_actor(Actor(state_dim, action_dim, actor_hidden_size, num_heads=num_agents))

        self.critic = DCCritic(state_dim, action_dim, num_bds, hidden_size=critic_hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def set_actor(self, actor):
        """Uses a copy of actor as the TD3 actor and target actor, with a fresh optimizer."""
        self.actor = copy.deepcopy(actor).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.total_it = 0

    def train_critic(self, policy, replay_buffers, num_steps):
        """
        Runs num_steps TD3 updates per agent, starting the TD3 actor from a copy of policy. The actor updates
        are only used to train the critic and are not kept.
        """
        self.set_actor(policy)  # doing this to update the target actor too
        for agent_id, replay_buffer in enumerate(replay_buffers):
            for _ in range(num_steps):
                self.train_step(replay_buffer, agent_id)

    def train_step(self, replay_buffer, agent_id):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, d_prime, d = [x.to(self.device) for x in replay_buffer.sample(self.batch_size)]

        # d holds the real descriptor, d_prime holds the sampled descriptor (could be negative or positive)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state, head=agent_id) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, d_prime)
            target_Q = torch.min(target_Q1, target_Q2)

            # Scale the reward by how close the expressed descriptor d is to the labeled descriptor d_prime
            similarity = torch.exp(-torch.norm(d_prime - d, dim=1, keepdim=True) / self.similarity_scale)

            scaled_reward = torch.where(reward < 0, reward * (1 + (1 - similarity)), reward * similarity)

            target_Q = scaled_reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, d_prime)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state, head=agent_id), d_prime).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def refine(self, policy, target_bd, replay_buffers):
        """
        Gradient ascent on the critic's Q1 conditioned on target_bd: refine_steps steps per agent, each on a
        batch of that agent's states. Updates policy in place and returns it.
        """
        actor_optim = torch.optim.Adam(policy.parameters(), lr=self.refine_lr)
        target_bds = torch.FloatTensor([target_bd] * self.refine_batch_size).to(self.device)

        # Freeze the critic so only the policy is updated
        for param in self.critic.parameters():
            param.requires_grad = False

        for _ in range(self.refine_steps):
            for agent_id, replay_buffer in enumerate(replay_buffers):
                state = replay_buffer.sample(self.refine_batch_size)[0].to(self.device)
                actor_loss = -self.critic.Q1(state, policy(state, head=agent_id), target_bds).mean()

                # Optimize the actor
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

        for param in self.critic.parameters():
            param.requires_grad = True

        return policy
