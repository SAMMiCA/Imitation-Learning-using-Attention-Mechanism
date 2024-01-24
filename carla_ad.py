'''
  ______   __     __  ________  __            __       
 /      \ /  |   /  |/        |/  |          /  |      
/$$$$$$  |$$ |   $$ |$$$$$$$$/ $$ |  ______  $$ |____  
$$ |__$$ |$$ |   $$ |$$ |__    $$ | /      \ $$      \ 
$$    $$ |$$  \ /$$/ $$    |   $$ | $$$$$$  |$$$$$$$  |
$$$$$$$$ | $$  /$$/  $$$$$/    $$ | /    $$ |$$ |  $$ |
$$ |  $$ |  $$ $$/   $$ |_____ $$ |/$$$$$$$ |$$ |__$$ |
$$ |  $$ |   $$$/    $$       |$$ |$$    $$ |$$    $$/ 
$$/   $$/     $/     $$$$$$$$/ $$/  $$$$$$$/ $$$$$$$/ 
'''

import argparse
import collections
import datetime
import time
import math
import os
import re
import sys
import weakref
import cv2
import torch
from collections import deque

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_r
    from pygame.locals import K_p
    from pygame.locals import K_1
    from pygame.locals import K_2
    from pygame.locals import K_3

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

import carla
from carla import ColorConverter as cc

from model.ADModel import ADModel
from model.ADAModel import ADAttentionModel

# from ModelPackage import ModelClass
global count
global control_count
global after_stuck
global is_start
is_start = False
after_stuck = False
count = 0
control_count = 0
global status_3_count, status_1_count, slow_count
status_3_count, status_1_count, slow_count = 0,0,0
class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self._world = carla_world
        try:
            self._map = self._world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self._hud = hud
        self._player = None
        self._camera_manager = None
        self._ad_status = 0 # 0: "Manual Driving", 1: "Collecting Data", 2: "Autonomous Driving"
        self.restart()
        self._world.on_tick(hud.on_world_tick)

    def restart(self):
        """Restart the world"""
        
        # Get the blueprint
        blueprint = self._world.get_blueprint_library().find(self._args.agent_vehicle_model)
        blueprint.set_attribute('role_name', 'hero')

        
        
        if self._player is not None:
            spawn_point = self._player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
        while self._player is None:
            if not self._map.get_spawn_points():
                time.sleep(1)
                continue
            spawn_points = self._map.get_spawn_points()
            # spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = carla.Transform(location=carla.Location(45.5,205.3,0.5),rotation=carla.Rotation(yaw=0))
            self._player = self._world.try_spawn_actor(blueprint, spawn_point)

            #
            vehicle_bp0 = self._world.get_blueprint_library().find('walker.pedestrian.0001')   
            vehicle_bp1 = self._world.get_blueprint_library().find('vehicle.mercedes.coupe_2020')   
            vehicle_bp2 = self._world.get_blueprint_library().find('vehicle.audi.etron')   
            vehicle_bp3 = self._world.get_blueprint_library().find('vehicle.tesla.cybertruck')   
            vehicle_bp4 = self._world.get_blueprint_library().find('vehicle.mercedes.coupe_2020')  
            # vehicle_bp5 = self._world.get_blueprint_library().find('vehicle.audi.etron') 
            
            vb0 = self._world.try_spawn_actor(vehicle_bp0, carla.Transform(location=carla.Location(168.3,179.5,2),rotation=carla.Rotation(yaw=-20)))
            vb1 = self._world.try_spawn_actor(vehicle_bp1, carla.Transform(location=carla.Location(203.2,89.9,0.5),rotation=carla.Rotation(yaw=-88)))
            vb2 = self._world.try_spawn_actor(vehicle_bp2, carla.Transform(location=carla.Location(207.6,-32.6,0.5),rotation=carla.Rotation(yaw=-72)))
            vb3 = self._world.try_spawn_actor(vehicle_bp3, carla.Transform(location=carla.Location(-127.0,-207.6,20),rotation=carla.Rotation(yaw=142)))
            vb4 = self._world.try_spawn_actor(vehicle_bp4, carla.Transform(location=carla.Location(-236.9,-132.3,20),rotation=carla.Rotation(yaw=111)))
            # vb5 = self._world.try_spawn_actor(vehicle_bp5, carla.Transform(location=carla.Location(-191.3,-190.1,20),rotation=carla.Rotation(yaw=28)))
        
        if self._args.sync:
            self._world.tick()
        else:
            self._world.wait_for_tick()

        # Set up the sensors
        
        self._camera_manager = CameraManager(self._player, self._hud, self)
        self._camera_manager.set_sensor()
        

        
    def tick(self, clock):
        self._hud.tick(self, clock)

    def render(self, display):
        self._camera_manager.render(display)
        self._hud.render(display)

    def destroy(self):
        sensors = [
            self._camera_manager._dp_camera,
            self._camera_manager._fv_camera,
            self._camera_manager._bev_camera,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.destroy()
                print(str(sensor) + " destroyed")
        if self._player is not None:
            self._player.destroy()
            print(str(self._player) + " destroyed")

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world):
        self._control = carla.VehicleControl()
        self._world = world
        self._reverse = False

    def parse_events(self, clock):
        global is_start
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYUP:
                if event.key == K_r:
                    self._reverse = not self._reverse
                    self._control.reverse = self._reverse
                if event.key == K_1:
                    self._world._ad_status = 0
                if event.key == K_2:
                    self._world._ad_status = 1
                if event.key == K_3:
                    self._world._ad_status = 2
                    is_start = True
                if self._is_quit_shortcut(event.key):
                    return None
        
        self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
        
        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.0)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._server_fps = 0
        self._frame = 0 
        self._simulation_time = 0
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self._server_fps = self._server_clock.get_fps()
        self._frame = timestamp.frame
        self._simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        transform = world._player.get_transform()
        vel = world._player.get_velocity()
        control = world._player.get_control()

        self._info_text = [
            'Server:  % 16.0f FPS' % self._server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self._simulation_time)),
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN}' % (transform.rotation.yaw),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
        ]

        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear),
                'AD Status: %s' % {0: 'Manual Driving', 1: 'Collecting Data', 2: 'Autonomous Driving'}.get(world._ad_status, world._ad_status),
            ]
        
    def render(self, display):
        info_surface = pygame.Surface((220, self.dim[1]))
        info_surface.set_alpha(100)
        display.blit(info_surface, (0, 0))
        v_offset = 4
        bar_h_offset = 100
        bar_width = 106
        for item in self._info_text:
            if v_offset + 18 > self.dim[1]:
                break
            if isinstance(item, tuple):
                if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    fig = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset + fig * (bar_width - 6), v_offset +8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                item = item[0]
            if item:
                surface = self._font_mono.render(item, True, (255, 255, 255))
                display.blit(surface, (8, v_offset))
            v_offset += 18

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, world):
        self._dp_camera = None
        self._fv_camera = None
        self._bev_camera = None
        self._fv_deque = deque(maxlen=10)
        self._bev_deque = deque(maxlen=10)
        self._camera_transform_deque = deque()
        self._steering_deque = deque()
        self._throttle_deque = deque()
        self._brake_deque = deque()
        self._velocity_deque = deque()
        self._surface = None
        self._parent_actor = parent_actor
        self._hud = hud
        self._world = world
        
        attachment = carla.AttachmentType
        self._dp_camera_transform = (carla.Transform(carla.Location(x=-10.5, z=4.5), carla.Rotation(pitch=-8.0)), attachment.Rigid)
        self._fv_camera_transform = (carla.Transform(carla.Location(x=3.0, z=1.5)), attachment.Rigid)
        self._bev_camera_transform = (carla.Transform(carla.Location(x=18.0, y=0.0, z=8.0), carla.Rotation(pitch=-90.0)), attachment.Rigid)
        
        self._dp_camera_info = ['sensor.camera.rgb', cc.Raw, 'Camera RGB']
        self._fv_camera_info = ['sensor.camera.rgb', cc.Raw, 'Camera RGB']
        self._bev_camera_info = ['sensor.camera.rgb', cc.Raw, 'Camera RGB']

        bp_library = self._parent_actor.get_world().get_blueprint_library()
        dp_camera_bp = bp_library.find(self._dp_camera_info[0])
        fv_camera_bp = bp_library.find(self._fv_camera_info[0])
        bev_camera_bp = bp_library.find(self._bev_camera_info[0])

        dp_camera_bp.set_attribute('image_size_x', str(hud.dim[0]))
        dp_camera_bp.set_attribute('image_size_y', str(hud.dim[1]))

        fv_camera_bp.set_attribute('image_size_x', '600')
        fv_camera_bp.set_attribute('image_size_y', '400')

        bev_camera_bp.set_attribute('image_size_x', '200')
        bev_camera_bp.set_attribute('image_size_y', '800')

        self._dp_camera_info.append(dp_camera_bp)
        self._fv_camera_info.append(fv_camera_bp)
        self._bev_camera_info.append(bev_camera_bp)

    def set_sensor(self):
        if self._dp_camera is not None:
            self._dp_camera.destroy()
            self._dp_camera = None
            self._surface = None

        if self._fv_camera is not None:
            self._fv_camera.destroy()
            self._fv_camera = None
            self._surface = None

        if self._bev_camera is not None:
            self._bev_camera.destroy()
            self._bev_camera = None
            self._surface = None

        self._dp_camera = self._parent_actor.get_world().spawn_actor(
            self._dp_camera_info[-1],
            self._dp_camera_transform[0],
            attach_to = self._parent_actor,
            attachment_type = self._dp_camera_transform[1])
        
        self._fv_camera = self._parent_actor.get_world().spawn_actor(
            self._fv_camera_info[-1],
            self._fv_camera_transform[0],
            attach_to = self._parent_actor,
            attachment_type = self._fv_camera_transform[1])
        
        self._bev_camera = self._parent_actor.get_world().spawn_actor(
            self._bev_camera_info[-1],
            self._bev_camera_transform[0],
            attach_to = self._parent_actor,
            attachment_type = self._bev_camera_transform[1])

        weak_self = weakref.ref(self)
        self._dp_camera.listen(lambda image: CameraManager._parse_image(weak_self, image, 'dp'))
        self._fv_camera.listen(lambda image: CameraManager._parse_image(weak_self, image, 'fv', data_deque=self._fv_deque))
        self._bev_camera.listen(lambda image: CameraManager._parse_image(weak_self, image, 'bev', data_deque=self._bev_deque))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, data, type, transform_deque=None, data_deque=None):
        self = weak_self()
        if not self:
            return

        if type == 'dp':
            data.convert(self._dp_camera_info[1])
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if type == 'fv':
            data.convert(self._fv_camera_info[1])
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            data_deque.append(array)

        if type == 'bev':
            data.convert(self._bev_camera_info[1])
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            data_deque.append(array)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    global count
    global status_3_count, status_1_count, slow_count
    global control_count
    global after_stuck
    global is_start
    stuck_record_count = 0
    lap_end_time = None
    print_status = True
    
    

    pygame.init()
    pygame.font.init()
    world = None
    actor_list = []
    try:
        if args.seed:
            np.random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        client.load_world('Town05')
        client.reload_world()


        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        
        clock = pygame.time.Clock()

        # collect image
        img_cnt= 0
        date = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        img_folder_front = os.path.join(os.getcwd(), f'fv_out', date)
        img_folder_bev = os.path.join(os.getcwd(), f'bev_out', date)


        # ADModel
        
        ad_model = None
        weight_path = "weights/latest.pt"
        if os.path.exists(weight_path):
            device = torch.device("cuda:1")
            # ad_model = ADModel().to(device)
            ad_model = ADAttentionModel().to(device)
            chkpt = torch.load("/home/ave/carla_ad_students_dh/weights/2023_12_20_02_29_45/latest.pt")
            ad_model.load_state_dict(chkpt['model'])
            ad_model.eval()
        CRASHdetector_blueprint = sim_world.get_blueprint_library().find('sensor.other.collision')
        global CRASHdetector
        crash_queue = deque()
        CRASHdetector = sim_world.spawn_actor(CRASHdetector_blueprint, carla.Transform(), attach_to = world._player)
        CRASHdetector.listen(crash_queue.append)
        actor_list.append(CRASHdetector)
        Start3 = False # REFEREE
        InitTime = 0 # REFEREE

        while True:
            if args.sync:
                world._world.tick()
            clock.tick_busy_loop(60)

            world.tick(clock)
            world.render(display)

            cm = world._camera_manager
            if cm._fv_deque.__len__() > 0 :
                fv_image = world._camera_manager._fv_deque.pop()
                if not (cm._bev_deque.__len__() > 0):
                    continue
                bev_image = world._camera_manager._bev_deque.pop()
                vel_vec = world._player.get_velocity()
                speed = math.sqrt(vel_vec.x**2 + vel_vec.y**2 + vel_vec.z**2)
                
                control = controller.parse_events(clock)
                control.brake = 0

                if (world._ad_status ==1):
                    if not os.path.exists(img_folder_front):
                        os.makedirs(img_folder_front)
                    if not os.path.exists(img_folder_bev):
                        os.makedirs(img_folder_bev)
                    img_cnt += 1

                    if img_cnt % 10 == 0:
                        pedal = control.throttle - control.brake
                        img_path_front = os.path.join(img_folder_front, f"{img_cnt:06d}_speed_{speed:03.2f}_pedal_{pedal:.2f}_steer_{control.steer:.2f}.png")
                        img_path_bev = os.path.join(img_folder_bev, f"{img_cnt:06d}_speed_{speed:03.2f}_pedal_{pedal:.2f}_steer_{control.steer:.2f}.png")
                        
                        cv2.imwrite(img_path_front, cv2.cvtColor(fv_image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(img_path_bev, cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                if (world._ad_status == 2) and (ad_model is not None):
                    if Start3 == False : # REFEREE 
                        Start3 = True # REFEREE
                        InitTime = snapshot.timestamp.elapsed_seconds # REFEREE
                    
                    _control = carla.VehicleControl()
                    fv_image = cv2.cvtColor(cv2.resize(fv_image.copy(), (150, 100)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    bev_image = cv2.cvtColor(cv2.resize(bev_image.copy(), (100, 400)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    # fv_image = cv2.resize(fv_image.copy(), [150, 100]).transpose(2, 0, 1)
                    pedal_pred, steer_pred = ad_model(torch.Tensor(fv_image).unsqueeze(0).to(device), \
                                                        torch.Tensor(bev_image).unsqueeze(0).to(device), \
                                                        torch.Tensor([speed]).unsqueeze(0).to(device))
                    pedal = pedal_pred.item() * 1.8
                    if speed > 28: pedal = 0.1
                    steer = steer_pred.item() * 0.77 #0.77
                    if abs(steer) > 0.03:
                        steer * 1.6
                    # print(control_count)
                    transform = world._player.get_transform() # REFEREE
                    if 18.5 < transform.location.x < 20.0 and 200 < transform.location.y < 215 and lap_end_time is None: # REFEREE
                        lap_end_time = time.time() # REFEREE
                        print("Lap Timer Ended") # REFEREE
                        break # REFEREE




                    try:
                        if (transform.location.x > -150 and transform.location.x < 100) and (transform.location.y > -203 and transform.location.y < -190): 
                            status_1 = True
                            factor = 0.7
                            status_1_max_count = 4
                            is_break_mode = False
                            if speed > 25: pedal = 0.1
                        elif (transform.location.x > -243.5 and transform.location.x < -230) and (transform.location.y > -110 and transform.location.y < 110): 
                            status_1 = True
                            factor = 0.7
                            status_1_max_count = 5
                            is_break_mode = False
                        else: status_1 = False


                        if ((transform.location.x > -170 and transform.location.x < 110) and (transform.location.y > 180 and transform.location.y < 206.5)) and control_count > 3000:
                            status_2 = True
                        else: status_2 = False


                        if (transform.location.x > -80 and transform.location.x < 100) and (transform.location.y < -210 and transform.location.y > -220): status_3 = True
                        else: status_3 = False


                        if (transform.location.x > -140 and transform.location.x < -115) and (transform.location.y < 211 and transform.location.y > 200): status_4 = True
                        else: status_4 = False


                    except: pass


                    try:
                        if status_1 and status_1_count < status_1_max_count:
                            status_1_count += 1
                            steer = steer + 0.05 * factor
                            if is_break_mode: control.brake = 0.5
                            # print('fucking')
                        elif status_1 and status_1_count == status_1_max_count:
                            # print('fucking_end')
                            status_1_count = 0
                            status_1 = False


                        if status_2 and slow_count < 20:
                            if transform.location.y < 203.5:
                                if slow_count < 4:
                                    steer = steer + 0.7
                                else: steer = steer + 0.02
                            else:
                                steer = steer + 0.02
                            slow_count += 1
                            # print('fucking_slow')
                        elif status_2 and slow_count == 20:
                            # print('fucking_slow_end')
                            slow_count = 0
                            status_2 = False


                        if status_3 and status_3_count < 3:
                            status_3_count += 1
                            steer = steer - 0.025
                            # print('fucking_inverse')
                        elif status_3 and status_3_count == 3:
                            # print('fucking_inverse_end')
                            status_3_count = 0
                            status_3 = False


                        if status_4:
                            # print('state_4')
                            steer = max(steer + 0.04, 0.0)
                    except: pass

                    # print(control_count)
                    try:
                        if ((pedal > 0.2 and speed < 1) and control_count > 2000) and control_count - stuck_record_count > 800:
                            is_stuck = True
                        else: is_stuck = False
                        
                        if is_stuck:
                            after_stuck = True
                            for i in range (200):
                                # print('stuck mode')
                                control.reverse = True
                                pedal = 0.6
                                steer = 0
                                control.throttle = pedal
                                control.brake = 0
                                control.steer = steer
                                world._player.apply_control(control)
                                snapshot = sim_world.get_snapshot()
                                pygame.display.flip()
                                if args.sync:
                                    world._world.tick()
                                clock.tick_busy_loop(60)
                                world.tick(clock)
                                world.render(display)
                                
                                #time.sleep(0.05)
                            is_stuck = False
                            control.reverse = False
                            stuck_record_count = control_count
                            
                        if after_stuck and control_count - stuck_record_count <= 800:
                            # print('after_stuck')
                            pedal = 0.2
                            steer = steer * 1.3

                            if control_count - stuck_record_count == 400: after_stuck = False
                    except: pass





                    if pedal < 0:
                        control.brake = -pedal
                        control.throttle = 0
                    else:
                        control.throttle = pedal
                    # _control.steer = steer
                    control.steer = steer
                    # control = _control
                
                world._player.apply_control(control)
            # print(len(crash_queue))
                if print_status: print(control.throttle, control.steer)

            
            snapshot = sim_world.get_snapshot()
            pygame.display.flip()
            if is_start: control_count += 1
            transform = world._player.get_transform() # REFEREE
            if 18.5 < transform.location.x < 20.0 and 200 < transform.location.y < 215 and lap_end_time is None: # REFEREE
                lap_end_time = time.time() # REFEREE
                print("Lap Timer Ended") # REFEREE
                print(len(crash_queue))
                print_status = False
                break # REFEREE

    finally:
        if world is not None:
            settings = world._world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world._world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            print('World will be destroyed in seconds...')

            with open('MO500_crash.txt', 'w') as f:
                for item in crash_queue:
                    f.write(f"{item}\n")
                f.write(f"Total time: {snapshot.timestamp.elapsed_seconds} seconds")
            world.destroy()

        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """Main function"""

    argparser = argparse.ArgumentParser(description='CARLA Autonomous Driving Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--agent_vehicle_model',
        default='vehicle.tesla.cybertruck',
        help='Actor model (default: "vehicle.tesla.cybertruck")')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.sync = False

    print(__doc__)
    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()







