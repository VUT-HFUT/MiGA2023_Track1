#fusion npy file and pkl file into one pkl file 

import pickle
import numpy as np

set='train'

data_dir='./data/imigue/task1/{}_data.npy'.format(set)
label_dir='./data/imigue/task1/{}_label.pkl'.format(set)

sample_data=np.load(data_dir)
q=open(label_dir,'rb')
sample_name, sample_label = pickle.load(q)

sample_name_list=[]
sample_annotation=[]

for i in range(len(sample_name)):
    sample_name_list.append(sample_name[i])
    temp={}
    temp['frame_dir']=sample_name[i]
    temp['label']=sample_label[i]
    temp['img_shape']=tuple((1080,1920))
    temp['original_shape']=tuple((1080,1920))
    temp['total_frames']=90
    keypoint=sample_data[i][:2].transpose(3,1,2,0)
    score=sample_data[i][2:3].reshape(1,90,22)
    temp['keypoint']=keypoint
    temp['keypoint_score']=score
    sample_annotation.append(temp)

final_data={}
final_data['split']={set:sample_name_list}
final_data['annotations']=sample_annotation


with open('./data/imigue_{}.pkl'.format(set), 'wb') as qqq:
    pickle.dump(final_data, qqq)