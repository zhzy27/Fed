class net:
    def recover_aggrevate(self, selected_client_ids):
        for client_id in selected_client_ids:
            model = self.client_models[client_id][1]
            for m in model.modules():
                if isinstance(m, MetaBasicBlock):
                    m.recover()
        n_selected_clients = len(selected_client_ids)
        weights = [1.0 / n_selected_clients for _ in range(n_selected_clients)]

        local_states = []
        for client_id in selected_client_ids:
            local_model = self.client_models[client_id][1]
            local_states.append(ModuleState(copy.deepcopy(local_model.state_dict())))

        model_state = local_states[0] * weights[0]
        
        for idx in range(1, len(local_states)):
            model_state += local_states[idx] * weights[idx]


        base_client_id = selected_client_ids[0]

        aggregated_model = copy.deepcopy(self.client_models[base_client_id][1])
        

        model_state.copy_to_module(aggregated_model)

        return aggregated_model