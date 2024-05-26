import optuna
import logging
from logging_utils import root_logger
import common
from common import model_op,service_info_dict,conf_and_serv_info,ip_range,reso_range,fps_range, edge_cloud_cut_range, cpu_range, mem_range
from kb_user_wzl import KnowledgeBaseUser
optuna.logging.set_verbosity(logging.WARNING)

class MacroPlanHelper():
    def __init__(self,conf_names=None,serv_names=None,service_info_list=None,rsc_constraint=None,user_constraint=None,rsc_upper_bound=None,rsc_down_bound=None,work_condition=None,portrait_info=None,bandwidth_dict=None,macro_plan_dict=None):
        self.conf_names = conf_names
        self.serv_names = serv_names
        self.service_info_list = service_info_list
        self.rsc_constraint = rsc_constraint
        self.user_constraint = user_constraint
        self.rsc_upper_bound = rsc_upper_bound
        self.rsc_down_bound = rsc_down_bound
        self.work_condition = work_condition
        self.portrait_info = portrait_info
        self.bandwidth_dict = bandwidth_dict
        self.macro_plan_dict = macro_plan_dict
        assert 'macro_plans' in self.macro_plan_dict
        self.macro_plan_num = len(self.macro_plan_dict['macro_plans'])
        assert self.macro_plan_num >= 1
        self.bayes_n_trials = 30  # 对于每一个宏观调度建议进行粗粒度贝叶斯优化采样的次数
    
    # 利用贝叶斯优化寻找最优的宏观调度计划时，使用的目标函数
    def objective(self, trial):
        ################## 1.选择一个宏观调度计划，并根据该计划限制各类可调参数范围 ##################
        macro_plan_index = trial.suggest_int('macro_plan_index', 0, self.macro_plan_num-1)
        macro_plan = self.macro_plan_dict['macro_plans'][macro_plan_index]
        
        #################### 1.1 获取历史调度策略 ####################
        # 从画像信息中提取调度策略而不是从云端保存的历史调度策略中获取，这是为了避免云边之间并发导致的策略不一致
        old_conf = self.portrait_info['exe_plan'][common.PLAN_KEY_VIDEO_CONF]
        old_flow_mapping = self.portrait_info['exe_plan'][common.PLAN_KEY_FLOW_MAPPING]
        old_resource_limit = self.portrait_info['exe_plan'][common.PLAN_KEY_RESOURCE_LIMIT]
        old_edge_cloud_cut_choice = len(self.serv_names)
        for serv in self.serv_names:
            if old_flow_mapping[serv]['node_role'] == 'cloud':
                old_edge_cloud_cut_choice = self.serv_names.index(serv)
                break
        
        #################### 1.2 根据宏观调度指导进行范围限制 ####################
        ########### 1.2.1 对帧率、分辨率范围进行限制 ###########
        old_fps_index = fps_range.index(old_conf['fps'])
        old_reso_index = reso_range.index(old_conf['reso'])
        if self.macro_plan_dict['conf_adjust_direction'] == 0:  # 若配置无需改变，则限定配置查找范围为当前配置
            new_fps_range = [old_conf['fps']]
            conf_and_serv_info['fps'] = new_fps_range
            
            new_reso_range = [old_conf['reso']]
            conf_and_serv_info['reso'] = new_reso_range
        
        else:  # 若配置需要改变，则依据宏观建议限定的上下界来限定范围
            adjust_str = str(macro_plan[0]) + '_' + str(macro_plan[1])
            if self.macro_plan_dict['conf_adjust_direction'] == 1:
                assert adjust_str in self.macro_plan_dict['conf_upper_bound']
                new_fps_bound, new_reso_bound = self.macro_plan_dict['conf_upper_bound'][adjust_str]
            else:
                assert adjust_str in self.macro_plan_dict['conf_lower_bound']
                new_fps_bound, new_reso_bound = self.macro_plan_dict['conf_lower_bound'][adjust_str]
            
            assert new_fps_bound in fps_range
            new_fps_index = fps_range.index(new_fps_bound)
            fps_min_index = min(old_fps_index, new_fps_index)
            fps_max_index = max(old_fps_index, new_fps_index)
            new_fps_range = fps_range[fps_min_index: fps_max_index+1]
            conf_and_serv_info['fps'] = new_fps_range
            
            assert new_reso_bound in reso_range
            new_reso_index = reso_range.index(new_reso_bound)
            reso_min_index = min(old_reso_index, new_reso_index)
            reso_max_index = max(old_reso_index, new_reso_index)
            new_reso_range = reso_range[reso_min_index: reso_max_index+1]
            conf_and_serv_info['reso'] = new_reso_range
            
        
        ########### 1.2.2 对云边协同方式进行限制 ###########
        new_edge_cloud_cut_range = []
        if macro_plan[2] == 0:
            new_edge_cloud_cut_range = [old_edge_cloud_cut_choice]
        elif macro_plan[2] == 1:
            if old_edge_cloud_cut_choice == 0:
                new_edge_cloud_cut_range = [0]
            else:
                new_edge_cloud_cut_range = [i for i in range(old_edge_cloud_cut_choice)]
        else:
            if old_edge_cloud_cut_choice == len(self.serv_names):
                new_edge_cloud_cut_range = [len(self.serv_names)]
            else:
                new_edge_cloud_cut_range = [i for i in range(old_edge_cloud_cut_choice + 1, len(self.serv_names) + 1)]
        conf_and_serv_info['edge_cloud_cut_point'] = new_edge_cloud_cut_range
        
        
        ########### 1.2.3 对资源调整范围进行限制 ###########
        for i in range(len(self.serv_names)):
            temp_str = self.serv_names[i] + '_cpu_util_limit'
            if macro_plan[3 + 2*i] == 2:
                conf_and_serv_info[temp_str] = cpu_range
            else:
                temp_cpu_limit = old_resource_limit[self.serv_names[i]]["cpu_util_limit"]
                if temp_cpu_limit == 1.0:  # 暂时认为只有在云端才会设置cpu利用率为1.0，因此在云端执行的服务CPU调整范围为[1.0]
                    conf_and_serv_info[temp_str] = [1.0]
                else:
                    temp_cpu_limit_index = cpu_range.index(temp_cpu_limit)
                    if macro_plan[3 + 2*i] == 0:
                        conf_and_serv_info[temp_str] = [temp_cpu_limit]
                    elif macro_plan[3 + 2*i] == 1:
                        if temp_cpu_limit_index == len(cpu_range) - 1:
                            conf_and_serv_info[temp_str] = [temp_cpu_limit]
                        else:
                            conf_and_serv_info[temp_str] = cpu_range[temp_cpu_limit_index+1: len(cpu_range)]
                    else:
                        if temp_cpu_limit_index == 0:
                            conf_and_serv_info[temp_str] = [temp_cpu_limit]
                        else:
                            conf_and_serv_info[temp_str] = cpu_range[0: temp_cpu_limit_index]

            temp_str = self.serv_names[i] + '_mem_util_limit'
            conf_and_serv_info[temp_str] = mem_range
    
        ################## 2.进行少量贝叶斯优化查找，判断在该调度计划下获得的各类指标 ##################
        ########### 2.1建立查表器 ###########
        plan_checker = KnowledgeBaseUser(conf_names=self.conf_names,
                                        serv_names=self.serv_names,
                                        service_info_list=self.service_info_list,
                                        user_constraint=self.user_constraint,
                                        rsc_constraint=self.rsc_constraint,
                                        rsc_upper_bound=self.rsc_upper_bound,
                                        rsc_down_bound=self.rsc_down_bound,
                                        work_condition=self.work_condition,
                                        portrait_info=self.portrait_info,
                                        bandwidth_dict=self.bandwidth_dict
                                        )
        
        params_in_acc_in_rsc_cons, params_in_acc_out_rsc_cons, params_out_acc_cons, next_plan = plan_checker.get_plan_in_cons_2(self.bayes_n_trials)
        
        ################## 3.求目标函数并返回 ##################
        ######### 3.1 求约束满足率 #########
        cons_satisfied_num = len(params_in_acc_in_rsc_cons)
        cons_unsatisfied_num = len(params_in_acc_out_rsc_cons) + len(params_out_acc_cons)
        assert (cons_satisfied_num + cons_unsatisfied_num) != 0
        if cons_satisfied_num == 0:
            root_logger.warning("In MacroPlanHelper, prev execute plan:{}, macro plan:{}, cons_satisfied_num = 0".format(self.portrait_info['exe_plan'], macro_plan))
        cons_satisfied_rate = cons_satisfied_num / (cons_satisfied_num + cons_unsatisfied_num)
        
        ######### 3.2 求最低时延 #########
        sorted_params_in_acc_in_rsc_cons = sorted(params_in_acc_in_rsc_cons, key=lambda item:(item['pred_delay_total']))
        minimum_delay = 10
        if len(sorted_params_in_acc_in_rsc_cons) > 0:
            minimum_delay = min(minimum_delay, sorted_params_in_acc_in_rsc_cons[0]['pred_delay_total'])
        
        ######### 3.3 求目标函数 #########
        # 将二者进行归一化
        cons_satisfied_rate_1 = cons_satisfied_rate / max(cons_satisfied_rate, minimum_delay)
        minimun_delay_1 = minimum_delay / max(cons_satisfied_rate, minimum_delay)
        
        optimize_obj = 0.7 * cons_satisfied_rate_1 - 0.3 * minimun_delay_1  # 约束满足率和最低时延的加权求和
        optimize_obj = -optimize_obj
        
        # 恢复修改之前的值
        conf_and_serv_info['fps'] = fps_range
        conf_and_serv_info['reso'] = reso_range
        conf_and_serv_info['edge_cloud_cut_point'] = edge_cloud_cut_range
        for i in range(len(self.serv_names)):
            temp_str = self.serv_names[i] + '_cpu_util_limit'
            conf_and_serv_info[temp_str] = cpu_range
            
            temp_str = self.serv_names[i] + '_mem_util_limit'
            conf_and_serv_info[temp_str] = mem_range
        
        return optimize_obj
    
    # 衡量某个宏观调度计划性能的函数
    def get_macro_plan_value(self, macro_plan_index):
        macro_plan = self.macro_plan_dict['macro_plans'][macro_plan_index]
        
        #################### 1.1 获取历史调度策略 ####################
        # 从画像信息中提取调度策略而不是从云端保存的历史调度策略中获取，这是为了避免云边之间并发导致的策略不一致
        old_conf = self.portrait_info['exe_plan'][common.PLAN_KEY_VIDEO_CONF]
        old_flow_mapping = self.portrait_info['exe_plan'][common.PLAN_KEY_FLOW_MAPPING]
        old_resource_limit = self.portrait_info['exe_plan'][common.PLAN_KEY_RESOURCE_LIMIT]
        old_edge_cloud_cut_choice = len(self.serv_names)
        for serv in self.serv_names:
            if old_flow_mapping[serv]['node_role'] == 'cloud':
                old_edge_cloud_cut_choice = self.serv_names.index(serv)
                break
        
        #################### 1.2 根据宏观调度指导进行范围限制 ####################
        ########### 1.2.1 对帧率、分辨率范围进行限制 ###########
        old_fps_index = fps_range.index(old_conf['fps'])
        old_reso_index = reso_range.index(old_conf['reso'])
        if self.macro_plan_dict['conf_adjust_direction'] == 0:  # 若配置无需改变，则限定配置查找范围为当前配置
            new_fps_range = [old_conf['fps']]
            conf_and_serv_info['fps'] = new_fps_range
            
            new_reso_range = [old_conf['reso']]
            conf_and_serv_info['reso'] = new_reso_range
        
        else:  # 若配置需要改变，则依据宏观建议限定的上下界来限定范围
            adjust_str = str(macro_plan[0]) + '_' + str(macro_plan[1])
            if self.macro_plan_dict['conf_adjust_direction'] == 1:
                assert adjust_str in self.macro_plan_dict['conf_upper_bound']
                new_fps_bound, new_reso_bound = self.macro_plan_dict['conf_upper_bound'][adjust_str]
            else:
                assert adjust_str in self.macro_plan_dict['conf_lower_bound']
                new_fps_bound, new_reso_bound = self.macro_plan_dict['conf_lower_bound'][adjust_str]
            
            assert new_fps_bound in fps_range
            new_fps_index = fps_range.index(new_fps_bound)
            fps_min_index = min(old_fps_index, new_fps_index)
            fps_max_index = max(old_fps_index, new_fps_index)
            new_fps_range = fps_range[fps_min_index: fps_max_index+1]
            conf_and_serv_info['fps'] = new_fps_range
            
            assert new_reso_bound in reso_range
            new_reso_index = reso_range.index(new_reso_bound)
            reso_min_index = min(old_reso_index, new_reso_index)
            reso_max_index = max(old_reso_index, new_reso_index)
            new_reso_range = reso_range[reso_min_index: reso_max_index+1]
            conf_and_serv_info['reso'] = new_reso_range
            
        
        ########### 1.2.2 对云边协同方式进行限制 ###########
        new_edge_cloud_cut_range = []
        if macro_plan[2] == 0:
            new_edge_cloud_cut_range = [old_edge_cloud_cut_choice]
        elif macro_plan[2] == 1:
            if old_edge_cloud_cut_choice == 0:
                new_edge_cloud_cut_range = [0]
            else:
                new_edge_cloud_cut_range = [i for i in range(old_edge_cloud_cut_choice)]
        else:
            if old_edge_cloud_cut_choice == len(self.serv_names):
                new_edge_cloud_cut_range = [len(self.serv_names)]
            else:
                new_edge_cloud_cut_range = [i for i in range(old_edge_cloud_cut_choice + 1, len(self.serv_names) + 1)]
        conf_and_serv_info['edge_cloud_cut_point'] = new_edge_cloud_cut_range
        
        
        ########### 1.2.3 对资源调整范围进行限制 ###########
        for i in range(len(self.serv_names)):
            temp_str = self.serv_names[i] + '_cpu_util_limit'
            if macro_plan[3 + 2*i] == 2:
                conf_and_serv_info[temp_str] = cpu_range
            else:
                temp_cpu_limit = old_resource_limit[self.serv_names[i]]["cpu_util_limit"]
                if temp_cpu_limit == 1.0:  # 暂时认为只有在云端才会设置cpu利用率为1.0，因此在云端执行的服务CPU调整范围为[1.0]
                    conf_and_serv_info[temp_str] = [1.0]
                else:
                    temp_cpu_limit_index = cpu_range.index(temp_cpu_limit)
                    if macro_plan[3 + 2*i] == 0:
                        conf_and_serv_info[temp_str] = [temp_cpu_limit]
                    elif macro_plan[3 + 2*i] == 1:
                        if temp_cpu_limit_index == len(cpu_range) - 1:
                            conf_and_serv_info[temp_str] = [temp_cpu_limit]
                        else:
                            conf_and_serv_info[temp_str] = cpu_range[temp_cpu_limit_index+1: len(cpu_range)]
                    else:
                        if temp_cpu_limit_index == 0:
                            conf_and_serv_info[temp_str] = [temp_cpu_limit]
                        else:
                            conf_and_serv_info[temp_str] = cpu_range[0: temp_cpu_limit_index]

            temp_str = self.serv_names[i] + '_mem_util_limit'
            conf_and_serv_info[temp_str] = mem_range
    
        ################## 2.进行少量贝叶斯优化查找，判断在该调度计划下获得的各类指标 ##################
        ########### 2.1建立查表器 ###########
        plan_checker = KnowledgeBaseUser(conf_names=self.conf_names,
                                        serv_names=self.serv_names,
                                        service_info_list=self.service_info_list,
                                        user_constraint=self.user_constraint,
                                        rsc_constraint=self.rsc_constraint,
                                        rsc_upper_bound=self.rsc_upper_bound,
                                        rsc_down_bound=self.rsc_down_bound,
                                        work_condition=self.work_condition,
                                        portrait_info=self.portrait_info,
                                        bandwidth_dict=self.bandwidth_dict
                                        )
        
        params_in_acc_in_rsc_cons, params_in_acc_out_rsc_cons, params_out_acc_cons, next_plan = plan_checker.get_plan_in_cons_2(self.bayes_n_trials)
        
        ################## 3.求目标函数并返回 ##################
        ######### 3.1 求约束满足率 #########
        cons_satisfied_num = len(params_in_acc_in_rsc_cons)
        cons_unsatisfied_num = len(params_in_acc_out_rsc_cons) + len(params_out_acc_cons)
        assert (cons_satisfied_num + cons_unsatisfied_num) != 0
        if cons_satisfied_num == 0:
            root_logger.warning("In MacroPlanHelper, prev execute plan:{}, macro plan:{}, cons_satisfied_num = 0".format(self.portrait_info['exe_plan'], macro_plan))
        cons_satisfied_rate = cons_satisfied_num / (cons_satisfied_num + cons_unsatisfied_num)
        
        ######### 3.2 求最低时延 #########
        sorted_params_in_acc_in_rsc_cons = sorted(params_in_acc_in_rsc_cons, key=lambda item:(item['pred_delay_total']))
        minimum_delay = 10
        if len(sorted_params_in_acc_in_rsc_cons) > 0:
            minimum_delay = min(minimum_delay, sorted_params_in_acc_in_rsc_cons[0]['pred_delay_total'])
        
        ######### 3.3 求目标函数 #########
        # 将二者进行归一化
        cons_satisfied_rate_1 = cons_satisfied_rate / max(cons_satisfied_rate, minimum_delay)
        minimun_delay_1 = minimum_delay / max(cons_satisfied_rate, minimum_delay)
        
        optimize_obj = 0.7 * cons_satisfied_rate_1 - 0.3 * minimun_delay_1  # 约束满足率和最低时延的加权求和
        optimize_obj = -optimize_obj
        
        # 恢复修改之前的值
        conf_and_serv_info['fps'] = fps_range
        conf_and_serv_info['reso'] = reso_range
        conf_and_serv_info['edge_cloud_cut_point'] = edge_cloud_cut_range
        for i in range(len(self.serv_names)):
            temp_str = self.serv_names[i] + '_cpu_util_limit'
            conf_and_serv_info[temp_str] = cpu_range
            
            temp_str = self.serv_names[i] + '_mem_util_limit'
            conf_and_serv_info[temp_str] = mem_range
        
        return optimize_obj
    
    # 利用贝叶斯优化确定最优的宏观调度计划
    def get_best_macro_plan(self):
        ### 1.利用贝叶斯优化选择最优的宏观计划
        study = optuna.create_study()
        trial_num = min(10, self.macro_plan_num)
        study.optimize(self.objective, n_trials=trial_num)
        
        trials = sorted(study.best_trials, key=lambda t: t.values)
        best_trial = trials[0]
        
        ### 2.返回最优宏观计划的相关信息
        best_macro_plan_index = best_trial.params['macro_plan_index']
        
        return best_macro_plan_index
    
    # 利用贝叶斯优化确定最优的k个宏观调度计划
    def get_top_k_macro_plans(self, k):
        ### 1.利用贝叶斯优化选择最优的宏观计划
        study = optuna.create_study()
        trial_num = min(10, self.macro_plan_num)
        study.optimize(self.objective, n_trials=trial_num)
        
        trials = sorted(study.best_trials, key=lambda t: t.values)
        
        ### 2.返回top-K宏观计划的相关信息
        top_k_index_list = []
        count = 0
        for trial in trials:
            if count < k:
                temp_index = trial.params['macro_plan_index']
                if temp_index not in top_k_index_list:  # 去除重复项
                    top_k_index_list.append(temp_index)
                    count += 1

        assert len(top_k_index_list) > 0
        return top_k_index_list

    # 遍历所有的宏观调度计划，并给出最优的k个
    def get_top_k_macro_plans_1(self, k):
        value_list = []
        
        # 遍历所有的宏观调度计划
        for index in range(self.macro_plan_num):
            temp_value = self.get_macro_plan_value(index)
            value_list.append({
                'index': index,
                'value': temp_value
            })
        
        sorted_value_list = sorted(value_list, key=lambda x: x['value'])
        
        top_k_index_list = [item['index'] for item in sorted_value_list[:k]]
        
        return top_k_index_list
        
        