import random





class ConstantDartsSpaceSampler():
    ''' Always returns the same set of random seeds to be used for 
        reproducible sampling of architectures from the DARTS search space'''
    def __init__(self):

        # CONSTANTS    
        self._SEED = 36
        self._MAXLEN = 1000

        self._random = random.Random(self._SEED)

        # generate MAXLEN number of random seeds
        population = range(self._MAXLEN * 1000000)
        self.choices = self._random.choices(population=population, k=self._MAXLEN)


    def get_archid(self, archid:int)->int:
        assert archid >= 0 and archid <= self._MAXLEN
        return self.choices[archid]




    

        