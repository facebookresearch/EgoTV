import os
import cv2
import json
import matplotlib.pyplot as plt


# traj = json.load(open('new_trajectories/clean_and_heat/Potato-None-None-13/trial_T20220907_072801_469546/traj_data.json'))
# img_folder = 'new_trajectories/clean_and_heat/Potato-None-None-13/trial_T20220907_072801_469546/raw_images'
# Egg-None-None-8/trial_T20220907_092753_177316
# traj = json.load(open('new_trajectories/clean_and_heat/TomatoSliced-None-None-16/trial_T20220907_090851_153008/traj_data.json'))
# img_folder = 'new_trajectories/clean_and_heat/TomatoSliced-None-None-16/trial_T20220907_090851_153008/raw_images'

traj = json.load(open('new_trajectories/clean_and_heat/Egg-None-None-8/trial_T20220907_092753_177316/traj_data.json'))
img_folder = 'new_trajectories/clean_and_heat/Egg-None-None-8/trial_T20220907_092753_177316/raw_images'

for img_dict in traj['images']:
    action_idx = img_dict['low_idx']
    high_pddl = img_dict['high_idx']
    before = img_dict['before']
    if action_idx == -1 or (action_idx == 0 and before == 'True'):
        obj_ind = [ind for ind, x in enumerate(traj['state_metadata'][0]) if x['objectType'] == 'Egg'][0]
        state = traj['state_metadata'][0][obj_ind]
    else:
        if before == 'True':
            obj_ind = [ind for ind, x in enumerate(traj['plan']['low_actions'][action_idx-1]['state_metadata'])
                       if x['objectType'] == 'Egg'][0]
            state = traj['plan']['low_actions'][action_idx-1]['state_metadata'][obj_ind]
        else: # after
            obj_ind = [ind for ind, x in enumerate(traj['plan']['low_actions'][action_idx]['state_metadata'])
                       if x['objectType'] == 'Egg'][0]
            state = traj['plan']['low_actions'][action_idx]['state_metadata'][obj_ind]
    img_file = os.path.join(img_folder, img_dict['image_name'])
    img = cv2.imread(img_file)
    result = img.copy()
    for key, box in img_dict['bbox'].items():
        object = key.split('|')[0]
        if object == 'Egg':
            if state['visible']:
                cv2.rectangle(result, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                ind = 0
                for state_k, state_v in state.items():
                    if state_k in ['visible', 'isHot', 'isClean', 'inReceptacle', 'holdsAny']:
                        ind += 1
                        if state_k == 'inReceptacle':
                            if state_v is None:
                                receps = 'None'
                            else:
                                receps = ', '.join([recep.split('|')[0] for recep in state_v])
                            cv2.putText(result, state_k + ': ' + receps, (int(box[0]), int(box[1]) - 30 * ind),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                        else:
                            cv2.putText(result, state_k + ': ' + str(state_v), (int(box[0]), int(box[1]) - 30 * ind),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            # ind += 1
            if action_idx != -1:
                cv2.putText(result, 'action: ' + traj['plan']['low_actions'][action_idx]['api_action']['action'],
                            (int(box[0]), int(box[1]) - 30 * 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                cv2.putText(result, 'action: ' + traj['plan']['high_pddl'][high_pddl]['discrete_action']['action'] +
                            '(' + traj['plan']['high_pddl'][high_pddl]['discrete_action']['args'][0] + ')',
                            (int(box[0]), int(box[1]) - 30 * 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        # elif object == 'StoveBurner':
        #     cv2.rectangle(result, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        #     cv2.putText(result, 'StoveBurner', (int(box[0]), int(box[1]) - 30 * ind),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # elif object == 'StoveKnob':
        #     cv2.rectangle(result, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
        #     cv2.putText(result, 'StoveKnob', (int(box[0]), int(box[1]) - 30 * ind),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        # elif object == 'SinkBasin':
        #     cv2.rectangle(result, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        #     cv2.putText(result, 'SinkBasin', (int(box[0]), int(box[1]) - 30 * ind),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            continue


        # plt.imsave(img_folder + '_bar/' + img_dict['image_name'], result)
        # plt.imshow(result)
        # plt.show()
        cv2.imwrite(img_folder + '_bar/' + img_dict['image_name'], result)