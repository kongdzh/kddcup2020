import pandas as pd
import numpy as np

click_columns = ['user_id','item_id','time']
train_click_T = pd.DataFrame()
test_click_T = pd.DataFrame()
test_qtime_T = pd.DataFrame()


# 第t个阶段，读取所有文件并合并
t = 4
for i in range(t):
    temp_train_click = pd.read_csv('underexpose_train/underexpose_train_click-%s.csv'%(str(i)),sep=',',names = click_columns )
    temp_test_click = pd.read_csv('underexpose_test/underexpose_test_click-%s.csv'%(str(i)),sep=',',names = click_columns)
    temp_test_qtime = pd.read_csv('underexpose_test/underexpose_test_qtime-%s.csv'%(str(i)),sep=',',names = ['user_id','time'])
    train_click_T = train_click_T.append(temp_train_click)
    test_click_T = test_click_T.append(temp_test_click)
    test_qtime_T = test_qtime_T.append(temp_test_qtime)

# 拼接所有历史点击数据
data_click_T = pd.concat([train_click_T,test_click_T],axis = 0).reset_index(drop = True)


# 计算所有商品的流行度-历史点击次数排序
data_click_T_item = data_click_T.groupby('item_id',as_index= False)['user_id'].agg({'user_dis':'unique','click_cnt':'count'})
data_click_T_item_rank = data_click_T_item.sort_values(by = 'click_cnt',axis = 0,ascending = False)['item_id'].tolist()


def next_item_by(user_list):
    '''
    针对所有用户，计算历史上某个商品点击之后下一个被点击商品的list
    返回数据格式：{'item_id':{'next_item_1':'click_count1','next_item2':'click_count2'}}
    '''
    item_seq={}
    for user_id in user_list:
        temp = data_click_T[data_click_T.user_id == user_id]
        temp_1 = temp.sort_values(by = 'time',axis = 0,ascending = True)
        c = temp_1['item_id'].tolist()
        for index,item in enumerate(c):
            if index ==len(c)-1:break
            if item not in item_seq:
                item_seq[item] = {}
                if c[index+1] not in item_seq[item]:
                    item_seq[item][c[index+1]] = 1
                else:
                    item_seq[item][c[index+1]] += 1
            else:    
                if c[index+1] not in item_seq[item]:
                    item_seq[item][c[index+1]] = 1
                else:
                    item_seq[item][c[index+1]] += 1
    return item_seq

item_seq = next_item_by(data_click_T['user_id'].unique())


def last_item(user_list):
    """
    获取测试集用户上一次点击的商品id
    返回数据格式：{'user_id':'last_click_item_id'}}
    """
    user_lastitem={}
    for user_id in user_list:
        temp =test_click_T[test_click_T.user_id == user_id]
        temp_1 = temp.sort_values(by = 'time',axis = 0,ascending = False).iloc[0:1,1:2].values[0][0]
        user_lastitem[user_id]= temp_1
    return user_lastitem

user_lastitem = last_item(test_click_T.user_id.tolist())


# 根据用户上一次点击的商品，把历史统计出来的点完这个商品还会点哪些的统计结果保存到对应user_id,并且把这个作为结果提交
user_nextitem_dict= {}
for user_id in user_lastitem.keys():
    item_id = user_lastitem[user_id]
    if item_id not in item_seq:
        user_nextitem_dict[user_id]= []
    else:
        next_item_dict = item_seq[item_id]
        temp =sorted(next_item_dict.items(),key = lambda x:x[1],reverse = True)
        next_item_list =[]
        for i in range(len(temp)):
            next_item_list.append(temp[i][0])
        user_nextitem_dict[user_id]= next_item_list



def deal(x):
    """对测试集的所有用户，用当前商品历史上下一次点击商品作为预测目标。如果长度不足50,则用所有商品流行度补全"""
    y=[]
    if len(user_nextitem_dict[x]) >=50:
        y = user_nextitem_dict[x][0:50]
    else:
        y=user_nextitem_dict[x]
        for i in data_click_T_item_rank:
            if i in y:continue
            y.append(i)
            if len(y)>=50:
                break
    return y
    
test_qtime_T['candidate'] = test_qtime_T['user_id'].apply(lambda x :deal(x))


"""生成对应的列"""
for i in range(50):
    if i < 9:
        col = 'item_id_0'+str(i+1)
    else:
        col = 'item_id_'+str(i+1)
    test_qtime_T[col] = test_qtime_T['candidate'].apply(lambda x: x[i])    

test_qtime_T = test_qtime_T.drop(columns=['time','candidate'])

test_qtime_T.to_csv('Result/underexpose_submit_T.csv',sep=',',index = False,header = False)