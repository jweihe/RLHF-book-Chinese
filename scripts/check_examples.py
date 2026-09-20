"""Execute the book's actual teaching snippets against small numerical fixtures.

Requires torch, numpy and jinja2. Run from the repository root.
"""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import re
import unittest

import jinja2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)


def snippet(chapter, marker, language="python"):
    path = next((ROOT / "chapters").glob(f"{chapter:02d}-*.md"))
    blocks = re.findall(rf"```{language}\n(.*?)```", path.read_text(), re.S)
    matches = [block for block in blocks if marker in block]
    if len(matches) != 1:
        raise ValueError((chapter, marker, len(matches)))
    return matches[0]


class BookExamples(unittest.TestCase):
    def test_kl_direction_and_mask(self):
        p = torch.tensor([0.8, 0.2])
        q = torch.tensor([0.4, 0.6])
        env = dict(logits=p.log().repeat(1, 2, 1),
                   ref_logits=q.log().repeat(1, 2, 1),
                   labels=torch.tensor([[0, 1]]),
                   completion_mask=torch.tensor([[1, 0]]))
        code = snippet(8, "log_target=True")
        exec(code, env)
        expected = (p * (p / q).log()).sum()
        self.assertAlmostEqual(env["conditional_kl"].item(), expected.item())
        self.assertAlmostEqual(env["sampled_kl"].item(), np.log(2))
        env["ref_logits"] = env["logits"].clone()
        exec(code, env)
        self.assertAlmostEqual(env["conditional_kl"].item(), 0)
        self.assertAlmostEqual(env["sampled_kl"].item(), 0)

    def test_ppo_shapes_returns_and_padding(self):
        new = torch.zeros(2, 3, requires_grad=True)
        values = torch.zeros(2, 3, requires_grad=True)
        env = dict(torch=torch, completion_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                   new_per_token_logps=new, per_token_logps=torch.zeros(2, 3),
                   gae_advantages=torch.tensor([[2., -1, 999], [1, -2, 3]]),
                   returns=torch.tensor([[2., 4, 999], [1, 2, 3]]), values=values,
                   self=SimpleNamespace(cliprange=0.2, vf_coef=0.5))
        exec(snippet(11, "valid = advantages"), env)
        self.assertEqual(env["per_token_loss"].shape, (2, 3))
        self.assertAlmostEqual(env["value_loss"].item(), 3.4)
        env["loss"].backward()
        self.assertEqual(new.grad[0, 2].item(), 0)
        self.assertEqual(values.grad[0, 2].item(), 0)
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertLess(values.grad[0, 0].item(), 0)

    def test_grpo_equal_rewards_and_learning_direction(self):
        new = torch.zeros(6, 2, requires_grad=True)
        env = dict(torch=torch, rewards=torch.tensor([1., 1, 1, 0, 0, 1]),
                   new_per_token_logps=new, per_token_logps=torch.zeros(6, 2),
                   per_token_kl=torch.zeros(6, 2), completion_mask=torch.ones(6, 2),
                   self=SimpleNamespace(num_generations=3, cliprange=0.2, beta=0.04))
        exec(snippet(11, "mean_grouped_rewards ="), env)
        self.assertTrue(torch.equal(env["advantages"][:3], torch.zeros(3, 1)))
        env["loss"].backward()
        self.assertTrue(torch.equal(new.grad[:3], torch.zeros(3, 2)))
        self.assertGreater(new.grad[3, 0].item(), 0)  # gradient descent lowers bad actions
        self.assertLess(new.grad[5, 0].item(), 0)

    def test_rloo_group_layout(self):
        rewards = torch.tensor([[1., 10], [2, 20], [3, 30]])
        env = dict(rlhf_reward=rewards.flatten(), rloo_k=3)
        exec(snippet(11, "baseline = (rlhf_reward.sum"), env)
        torch.testing.assert_close(env["advantages"].reshape(3, 2),
                                   1.5 * (rewards - rewards.mean(0)))

    def test_dpo_gradient_and_reference_detached(self):
        chosen = torch.tensor([-2.], requires_grad=True)
        rejected = torch.tensor([-3.], requires_grad=True)
        env = dict(policy_chosen_logps=chosen, policy_rejected_logps=rejected,
                   reference_chosen_logps=torch.tensor([-2.]),
                   reference_rejected_logps=torch.tensor([-3.]), beta=0.2)
        exec(snippet(12, "pi_logratios ="), env)
        self.assertAlmostEqual(env["losses"].item(), np.log(2))
        env["losses"].sum().backward()
        self.assertAlmostEqual(chosen.grad.item(), -0.1)
        self.assertAlmostEqual(rejected.grad.item(), 0.1)
        self.assertFalse(env["chosen_rewards"].requires_grad)

    def test_aggregation_independent_gradients(self):
        env = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(snippet(11, "def masked_mean"), env)
        # Last backward must not contain gradients accumulated by earlier examples.
        torch.testing.assert_close(env["ratio"].grad, env["masks"] * (2 / 11))

    def test_rejection_sampling_indices(self):
        rewards = np.array([[.7,.3,.5,.2], [.4,.8,.6,.5], [.9,.3,.4,.7],
                            [.2,.5,.8,.6], [.5,.4,.3,.6]])
        chapter = (ROOT / "chapters/10-rejection-sampling.md").read_text()
        listed = re.search(r'S_5\(R_\{flat\}\) = \[(.*?)\]', chapter).group(1)
        indices = [int(x) for x in listed.split(',')]
        self.assertEqual(set(indices), set(np.argsort(rewards.ravel())[-5:]))
        self.assertEqual(np.argmax(rewards, axis=1).tolist(), [0, 1, 0, 2, 3])

    def test_chat_roles(self):
        template = jinja2.Environment().from_string(snippet(9, "set offset", "jinja"))
        def fail(message):
            raise ValueError(message)
        def render(roles):
            return template.render(messages=[dict(role=r, content=r) for r in roles],
                                   bos_token="", add_generation_prompt=True,
                                   raise_exception=fail)
        for roles in ([], ["user"], ["system", "user"], ["user", "assistant", "user"]):
            self.assertIn("<|im_start|>assistant", render(roles))
        for roles in (["assistant"], ["system", "assistant"], ["user", "user"], ["user", "tool"]):
            with self.assertRaises(ValueError):
                render(roles)


if __name__ == "__main__":
    unittest.main(verbosity=2)
