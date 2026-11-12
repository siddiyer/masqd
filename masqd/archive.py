import random


class BehaviorArchive:
    """
    Archive of team policies keyed by behavior descriptor. Each descriptor (cell) keeps the
    highest-fitness policy found for it.
    """

    def __init__(self):
        self.archive_dict = {}  # descriptor -> (fitness, policy)
        self.behavior_desc_list = []  # descriptors in insertion order, used for sampling

    def add(self, policy, bd, fitness):
        if bd in self.archive_dict:
            # Only replace the stored policy if the new one is better
            if self.archive_dict[bd][0] < fitness:
                self.archive_dict[bd] = (fitness, policy)
        else:
            self.archive_dict[bd] = (fitness, policy)
            self.behavior_desc_list.append(bd)

    def sample_uniform(self):
        """Returns a (policy, descriptor) pair chosen uniformly at random from the filled cells."""
        bd = random.sample(self.behavior_desc_list, 1)[0]
        return self.archive_dict[bd][1], bd

    def qd_score(self):
        """Sum of the fitnesses of all policies in the archive."""
        score = 0
        for fitness, _ in self.archive_dict.values():
            score += fitness
        return score

    def max_fitness(self):
        return max((fitness for fitness, _ in self.archive_dict.values()), default=float("-inf"))

    def __len__(self):
        return len(self.archive_dict)
