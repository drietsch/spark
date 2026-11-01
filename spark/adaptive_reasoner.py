"""M17: learned memory routing feeds a live neural mathematical primitive.

No direct neural-function labels or write-instruction labels are given to either
learner. Only delayed query targets are supplied; scoring is prequential.
"""
from __future__ import annotations
from dataclasses import dataclass
from spark.learned_memory import MemoryRouter, Event
from spark.plastic_function import PlasticFunction


@dataclass
class AdaptiveReasoner:
    router: MemoryRouter
    function: PlasticFunction

    def initial_memory(self):return [0.]*self.router.keys

    def step(self, memory, event: Event, context: float, *, target=None, lr=0.06):
        """Prequential: output before online feedback, otherwise learn gate policy fixed."""
        if event.query:
            v=memory[event.key]
            z=(v,float(context),v*float(context))
            q,cache=self.function.forward(z)
            prediction=v+q
            if target is not None:
                self.function.update(z,prediction-target,lr,cache)
            return prediction
        if self.router.score(event)>.5: memory[event.key]=event.value
        return None
