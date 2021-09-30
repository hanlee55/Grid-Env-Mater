import os
import time
import numpy as np
import random
import copy

np.random.seed(1)


class Env:
    def __init__(self, render_speed=0.01, *args, **kwargs):
        self.render_speed = render_speed
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)

        self.agent = {"pos": [0, 0]}

         # 환경 환경설정 부분
        self.map_size = [5, 6]
        self.type_count = 2

        self.base_map = self.create_base_map()
        # self.map = [] 출력할때만 필요해서 삭제

        self.counter = 0
        self.rewards = []
        self.objects = []
        self.goal = []

        # 환경변수_불러오기
        self.env_var = kwargs['env_var']
        '''
        step_deduction: 스텝 소득 없을시 감점사항
        objects_point_add: object 가점 사항
        end_point_add: 종료지점 가점 사항
        '''

        self.checkPoint = False



    # new methods
    def check_if_reward(self):
        check_list = dict()
        check_list['if_goal'] = False

        rewards = 0

        for i, obj in enumerate(self.objects):
            if obj['pos'] == self.agent['pos'] and obj['type'] == 0:
                if not self.checkPoint:
                    # rewards += reward['reward']
                    rewards += self.env_var['objects_point_add']
                    self.checkPoint = True
        if self.checkPoint and self.agent['pos'][1] == 5:
            rewards += self.env_var['end_point_add']
            check_list['if_goal'] = True
        if rewards == 0:
            rewards = self.env_var['step_deduction']

        check_list['rewards'] = rewards

        return check_list

    def place_objects(self, count, work_range, except_position=[], self_place=[], types_count=[]):
        """# 가능한 y범위(세로) x범위(가로)+ 제외할 x,y 좌표 리스트
        # 랜덤 개수범위입력 -> 랜덤 배치
        # 수동배치도 가능"""

        result = []
        buffer = [i+1 for i in range(work_range[0]*work_range[1])]
        # 수동 추가
        if len(self_place):
            result += self_place
        # 추첨 제외
        for i in (self_place+except_position):
            if not(i[0]+1 > work_range[0] or i[1]+1 > work_range[1]):
                buffer.pop(buffer.index(work_range[0]*i[0]+i[1]+1))
        # 추첨
        if len(buffer) < count:
            count = len(buffer)
        result_buffer = random.sample(buffer, count)
        type_que = [0, 0]
        for i in result_buffer:
            if type_que[0]+1 != self.type_count:
                type_temp = type_que[0]
                type_que[1] += 1
                if types_count[type_que[0]] == type_que[1]:
                    type_que[0] += 1
                    type_que[1] = 0
            else:
                type_temp = self.type_count-1
            result.append({
                "pos": [(i - 1) // work_range[0], (i - 1) % work_range[0]],
                "type": type_temp
            })

        return result  # Example[{pos: [0, 1]}, {pos: [2, 15]}, {pos: [20, 13]}]

    def process_env(self):
        pass


    def reset(self):
        self.agent = {"pos": [0, 0]}
        self.checkPoint = False
        """
        보상 지점 랜덤 설정 방법: 각 번호에 리스트를 만들어서 리스트에서 뽑고 리스트에서 제거
        """
        objects_count = random.randint(1, 25)
        self.objects = self.place_objects(
            objects_count, [5, 5], except_position=[], self_place=[],
            types_count=[objects_count//2]
        )
        # self.reset_reward()

        return self.get_state()


    def step(self, action):
        self.counter += 1

        self.process_env()

        self.agent['pos'] = self.move_agent(self.agent['pos'], action)
        check = self.check_if_reward()
        done = check['if_goal']
        reward = check['rewards']

        s_ = self.get_state()

        return s_, reward, done

    # 반환할 상태에 맞게 배열 초기화
    def create_base_map(self):
        state_map = []  # [agent_pos, objects_map]

        # agnet_pos
        state_map.append([0, 0])  # agent position [y/map_size_y, x/map_size_x]

        # objects map
        state_map.append([])
        for i in range(self.map_size[0]):
            buffer = []
            for j in range(self.map_size[1]):
                buffer.append([0 for i in range(self.type_count)])
            state_map[1].append(buffer)

        return state_map

    def render(self):
        state_map = copy.deepcopy(self.base_map)
        # agent 배치
        state_map[0][0] = self.agent['pos'][0] / (self.map_size[0]-1)
        state_map[0][1] = self.agent['pos'][1] / (self.map_size[1]-1)

        # objects 배치
        for obj in self.objects:
            y = obj['pos'][0]  # 세로
            x = obj['pos'][1]  # 가로
            state_map[1][y][x][obj['type']] = 1

        return state_map

    def get_state(self):  # map 에 속성값이 들어가 있는경우 반환할 상태로 변환처리
        # NOTE 속성으로 좌표를 찍어둔 다음 state 출력할때 위치를 포함하기
        states = self.render()

        return states

    # TODO def 환경 처리 함수 (ex - 매번 환경이 움직이는 경우 이것을 처리)

    def move_agent(self, pos, action):  # pos = [0, 0]
        if action == 0:  # 상
            if pos[0] > 0:
                pos[0] -= 1
        elif action == 1:  # 하
            if pos[0] < self.map_size[0]-1:
                pos[0] += 1
        elif action == 2:  # 우
            if pos[1] < self.map_size[1] - 1:
                pos[1] += 1
        elif action == 3:  # 좌
            if pos[1] > 0:
                pos[1] -= 1

        return pos

    # 규칙에 맞게 시각화 구축
    def visualization(self, state):
        for i in state[1]:
            for j in i:
                print("[{}, {}]".format(j[0], j[1]), end=" ")
            print()

if __name__ == "__main__":
    test_env = Env(env_var={
        "step_deduction": -0.0001,
        "objects_point_add": 1,
        "end_point_add": 1
    })
    test_env.reset(
