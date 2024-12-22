from transformers.trainer import *
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict


class BiTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):


        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        self.model.save(output_dir)

        # if is_deepspeed_zero3_enabled():
        #     print('太好了呀！！！！！！！！！！！！！！！！！！！！！ zero3启动了')
        #     if state_dict is None:
        #         state_dict = self.model.state_dict()
        #     prefix = 'model.'
        #     assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        #     state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        #     lora_state_dict = get_peft_model_state_dict(self.model.model, state_dict)
        #     print(list(lora_state_dict.keys()))
        #     if self.args.process_index <= 0:
        #         torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
        #         print(f"Save adapter model at {output_dir}")


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss