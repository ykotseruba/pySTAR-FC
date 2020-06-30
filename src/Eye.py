import numpy as np

class Eye:
    def __init__(self, settings, env):
        self.settings = settings
        self.gazeCoords = -np.ones((1, 2), dtype=np.int32)
        self.foveate = settings.foveate
        self.env = env
        self.height = env.height
        self.width = env.width
        self.view = None
        self.viewFov = None

        if self.foveate:
            #from Foveate_GP_OGL import Foveate_GP_OGL
            #self.fov = Foveate_GP_OGL(dotPitch = -1, viewDist = settings.viewDist)

            #uncomment to use the pyCUDA code
            from Foveate import Foveate
            self.fov = Foveate(self.env.dotPitch, self.settings.viewDist, self.settings.rodsAndCones)

    def reset(self):
        self.height = self.env.getHeight()
        self.width = self.env.getWidth()
        self.gazeCoords = np.zeros((1, 2), dtype=np.int32)
        self.view = None
        self.viewFov = None

    def viewScene(self):
        # if gazeCoords are not initialized (i.e. equal to [-1,1]) automatically set the first
        # fixation at the center of the image
        if np.all(np.equal(self.gazeCoords, [-1, -1])):
            self.gazeCoords = np.array([self.height/2, self.width/2], dtype=np.int32)

        self.view = self.env.getEyeView(self.gazeCoords)
        self.foveateView()

    def foveateView(self):

        if self.foveate:
            self.fov.dotPitch = self.env.dotPitch
            self.fov.foveate(self.view, np.array([self.height/2, self.width/2], dtype=np.int32))
            self.viewFov = self.fov.imgFov

        else:
            self.viewFov = self.view

    def setGazeCoords(self, newGazeDirection):
        self.gazeCoords[0] = int(max(0, min(self.height-1, self.gazeCoords[0] + newGazeDirection[0])))
        self.gazeCoords[1] = int(max(0, min(self.width-1, self.gazeCoords[1] + newGazeDirection[1])))
