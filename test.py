class net:
    def recover_aggrevate(self, selected_client_ids):

        for client_id in selected_client_ids:
            model = self.client_models[client_id][1]
            for m in model.modules():
                if isinstance(m, MetaBasicBlock):
                    m.recover()

        base_client_id = selected_client_ids[0]
        base_model = self.client_models[base_client_id][1]
        global_params = copy.deepcopy(base_model.state_dict())
        for key in global_params:
            global_params[key].zero_()

        # --- 第三步：累加所有模型的参数 ---
        for client_id in selected_client_ids:
            model = self.client_models[client_id][1]
            local_params = model.state_dict()
            
            for key in global_params:

                global_params[key] += local_params[key]

        # --- 第四步：取平均 ---
        num_models = len(selected_client_ids)
        for key in global_params:
            if global_params[key].is_floating_point():
                global_params[key] /= num_models
            else:
                val_float = global_params[key].float() / num_models
                if global_params[key].is_floating_point():
                    global_params[key].copy_(val_float)
                else:
                    global_params[key].copy_(torch.round(val_float).long())
        aggregated_model = copy.deepcopy(base_model)
        aggregated_model.load_state_dict(global_params)

        return aggregated_model



