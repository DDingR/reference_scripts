import matlab.engine

class SimManager:
    def __init__(self, modelName) -> None:
        self.modelName = modelName

    def __get_obs(self, obsInfo):
        tout = [self.eng.workspace['tout']]

        obs = []
        for name in obsInfo:
            obs.append(self.eng.workspace[name])
        
        return tout, obs
    
    def __setParameter(self, block, name, value):
        self.eng.set_param(
            '{}/{}'.format(self.modelName, block),
            name, str(value),
            nargout=0
        )

    def connectMatlab(self):
        print('[ MNG] Connecting Matlab')
        self.eng = matlab.engine.start_matlab()
        print('[ MNG] Connected Successfully')

        print('[ MNG] Loading Simulink Model')
        self.eng.eval(
            "model = '{}';".format(self.modelName), nargout=0
        )
        self.eng.eval(
            "load_system(model)", nargout=0
        )
        print('[ MNG] Loaded Successfully')

    def reset(self, obsInfo, initial_parameters):
        # print('[ MNG] Reset Env')
        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)

        for block in initial_parameters:
            for name in initial_parameters[block]:
                value = initial_parameters[block][name]
                self.__setParameter(block, name, value)

        self.eng.set_param(self.modelName,
                           'SimulationCommand', 'start', 'SimulationCommand', 'pause', 
                           nargout=0)
        
        return self.__get_obs(obsInfo)

    def step(self, obsInfo, abj_parameters):
        for block in abj_parameters:
            for name in abj_parameters[block]:
                value = abj_parameters[block][name]
                self.__setParameter(block, name, value)

        self.eng.set_param(self.modelName,
                           'SimulationCommand', 'continue', 'SimulationCommand', 'pause', 
                           nargout=0)
        
        return self.__get_obs(obsInfo)

    def disconnectMatlab(self):
        print('[ MNG] Disconnecting Env')

        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()

        print('[ MNG] Disconnected Successfully')
