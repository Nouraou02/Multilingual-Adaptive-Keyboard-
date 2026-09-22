import torch
import torch.nn.functional as F  
import re    

class MultilingualKeyboardEngine:
    def __init__(self, base_model, base_model_router, tokenizer, router, device):
        self.model = base_model
        self.model_router = base_model_router
        self.tokenizer = tokenizer
        self.router = router.to(device).float()
        self.device = device
        self.mapping = {0: "base", 1: "fr_adapter", 2: "zh_adapter"}

    def get_next_words(self, text, top_k=15):
        # --- 1. GLOBAL SAFETY CHECK FOR EMPTY TEXT ---
        if not text or not text.strip():
            return [], "EN", 1.0

        try:
            self.router.eval()
            self.model.eval()
            
            is_at_space = text.endswith(" ")
            # Add leading space for model consistency
            context_text = text if text.startswith(" ") else " " + text
            
            inputs = self.tokenizer(context_text, return_tensors="pt", add_special_tokens=True).to(self.device)
            
            # --- 2. ROUTING ---
            with torch.no_grad():
                outputs = self.model_router(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1][:, -1, :].float()  
                logits_router = self.router(embedding)
                probs_router = F.softmax(logits_router, dim=1)
                conf, pred_idx = torch.max(probs_router, dim=1)
                conf, pred_idx = conf.item(), pred_idx.item()

                if pred_idx == 2 and not re.search(r'[\u4e00-\u9fff]', text):
                    pred_idx = 0
            
            target_adapter = self.mapping[pred_idx]
            display_label = {"base": "EN", "fr_adapter": "FR", "zh_adapter": "ZH"}.get(target_adapter, "EN")

            if target_adapter == "base":
                self.model.disable_adapter_layers()
            else:
                self.model.set_adapter(target_adapter)
                self.model.enable_adapter_layers()

            # --- 3. PREDICTION ---
            predictions = []
            seen_words = set()

            with torch.no_grad():
                model_outputs = self.model(**inputs)
                
                # Strict temp (0.2) for mid-word, standard (0.6) for next-word
                temp = 0.6 if is_at_space else 0.2
                next_token_logits = model_outputs.logits[:, -1, :] / temp
                
                # SAFE FRAGMENT EXTRACTION
                current_fragment = ""
                if not is_at_space:
                    words = text.split()
                    if words:
                        current_fragment = words[-1].lower()

                top_indices = torch.topk(next_token_logits, k=100, dim=-1).indices[0]
                
                for token_id in top_indices:
                    raw_token = self.tokenizer.decode([token_id])
                    word = raw_token.strip()
                    
                    if not word or len(word) < 1: continue
                    if not any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in word): continue

                    # --- CHINESE CLEANUP & COMPLETION ONLY LOGIC ---
                    if re.search(r'[\u4e00-\u9fff]', word):
                        # Strip entire context input if it is baked into the huge BPE token
                        if word.startswith(text):
                            word = word[len(text):]
                        elif current_fragment and word.startswith(current_fragment):
                            word = word[len(current_fragment):]
                        
                        word = word.strip()
                        # Skip if filtering leaves us empty or if we've already found this exact candidate
                        if not word or word in seen_words: continue
                        
                        # Add pure prediction tail and bypass EN/FR fragment combinations
                        predictions.append(word)
                        seen_words.add(word)
                        
                        if len(predictions) == 3: break
                        continue

                    # --- ENGLISH / FRENCH WORD FINISHING LOGIC ---
                    word_clean = word.lower()
                    if word_clean in seen_words: continue

                    if not is_at_space and current_fragment:
                        if word_clean.startswith(current_fragment) and word_clean != current_fragment:
                            predictions.append(word)
                            seen_words.add(word_clean)
                        elif not raw_token.startswith(" "):
                            combined = current_fragment + word_clean
                            if combined not in seen_words:
                                predictions.append(combined)
                                seen_words.add(word_clean)
                    else:
                        predictions.append(word)
                        seen_words.add(word_clean)
                    
                    if len(predictions) == 3: break
                        
            return predictions, display_label, conf

        except Exception as e:
            print(f"Error in engine: \(e)")
            return [], "EN", 0.0