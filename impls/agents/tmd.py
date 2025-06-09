from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    DiscreteStateActionRepresentation,
    GCActor,
    GCDiscreteActor,
    Param,
    StateRepresentation,
)


class TMDAgent(flax.struct.PyTreeNode):
    """Temporal Metric Distillation (TMD) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def mrn_distance_component(self, x, y):
        eps = 1e-6
        d = x.shape[-1]
        x_prefix = x[..., : d // 2]
        x_suffix = x[..., d // 2 :]
        y_prefix = y[..., : d // 2]
        y_suffix = y[..., d // 2 :]
        max_component = jnp.max(jax.nn.relu(x_prefix - y_prefix), axis=-1)
        l2_component = jnp.sqrt(jnp.square(x_suffix - y_suffix).sum(axis=-1) + eps)
        assert max_component.shape == l2_component.shape
        return max_component + l2_component

    @jax.jit
    def mrn_distance(self, x, y):
        K = self.config['components']

        x_split = jnp.stack(jnp.array_split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.array_split(y, K, axis=-1), axis=-1)
        dists = [self.mrn_distance_component(x_split[..., i], y_split[..., i]) for i in range(K)]

        return jnp.stack(dists, axis=-1).mean(axis=-1)

    def iqe_distance(self, x, y):
        k = self.config['components']  
        alpha_raw = self.network.select('alpha_raw')()
        alpha = jax.nn.sigmoid(alpha_raw)
        reshape = (x.shape[-1] // k, k)
        x = jnp.reshape(x, (*x.shape[:-1], *reshape))
        y = jnp.reshape(y, (*y.shape[:-1], *reshape))
        valid = x < y
        D = x.shape[-1]
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(ixy < D, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = (neg_inp_copies < 0) * (-1.0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(-1)
        result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
        return result

    @jax.jit
    def distance(self, x, y):
        x, y = jnp.broadcast_arrays(x, y)
        if self.config['use_iqe']:
            return self.iqe_distance(x, y)
        else:
            return self.mrn_distance(x, y)

    @jax.jit
    def contrastive_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]

        phi = self.network.select('phi')(batch['observations'], batch['actions'], params=grad_params)
        psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
        psi_next = self.network.select('psi')(batch['next_observations'], params=grad_params)
        psi_g = self.network.select('psi')(batch['value_goals'], params=grad_params)

        if len(phi.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]

        dist = self.distance(phi[:, :, None], psi_g[:, None, :])
        logits = -dist / jnp.sqrt(phi.shape[-1])
        # logits.shape is (e, B, B) with one term for positive pair and (B - 1) terms for negative pairs in each row.

        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)
        action_dist = self.distance(psi_s, phi)

        action_invariance_loss = jnp.mean(action_dist)

        dist_next = self.distance(psi_next[:, :, None], psi_g[:, None, :])

        t = self.config['t']
        gamma = self.config['discount']
        if self.config['stopgrad_psi_backup']:
            dist = self.distance(phi[:, :, None], jax.lax.stop_gradient(psi_g[:, None, :]))
        dist_next = jax.lax.stop_gradient(dist_next)

        delta = dist - dist_next
        mask = delta > t
        delta = jnp.where(mask, t, delta)
        divergence = gamma * jnp.exp(delta) - dist
        dw = self.config['diag_backup']
        divergence = divergence * (1 - dw) + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw

        divergence = jnp.clip(divergence, None, self.config['t'])
        backup_loss = jnp.mean(divergence)

        critic_loss = (
            contrastive_loss
            + self.config['zeta'] * action_invariance_loss
            + self.config['zeta'] * backup_loss
        )

        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, {
            'critic_loss': critic_loss,
            'action_invariance_loss': action_invariance_loss,
            'backup_loss': backup_loss,
            'contrastive_loss': contrastive_loss,
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
            'dist': dist.mean(),
            'biggest_diff_in_dist': jnp.max(dist - dist_next),
        }

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):
        # Maximize log Q if actor_log_q is True (which is default).

        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        # actions_roll = jnp.roll(batch['actions'], shift=1, axis=0)
        # assert q_actions.shape == actions_roll.shape

        phi = self.network.select('phi')(batch['observations'], q_actions)
        psi = self.network.select('psi')(batch['actor_goals'])
        q1, q2 = -self.distance(phi, psi)
        q = jnp.minimum(q1, q2)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])

        bc_loss = -(self.config['alpha'] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['state'] = encoder_module()
        if config['discrete']:
            phi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            psi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            phi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
            )
            psi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
            )
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
        if config['use_iqe']:
            network_info = dict(
                actor=(actor_def, (ex_observations, ex_goals)),
                phi=(phi_def, (ex_observations, ex_actions)),
                psi=(psi_def, (ex_goals,)),
                alpha_raw=(Param(), ()),
            )
        else:
            network_info = dict(
                actor=(actor_def, (ex_observations, ex_goals)),
                phi=(phi_def, (ex_observations, ex_actions)),
                psi=(psi_def, (ex_goals,)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='tmd',  # Agent name.
            lr=3e-4,
            components=2,  # Number of components in MRN.
            batch_size=512,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            zeta=0.2,
            t=5.0,
            diag_backup=0.5,  # Whether to use diagonal backup.
            stopgrad_psi_backup=True,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            use_iqe=False,  # Whether to use IQE distance or MRN distance
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
