         self.enable_contact_matching = enable_contact_matching
         self.reduce_contacts = reduce_contacts
         self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
         if broad_phase_mode == BroadPhaseMode.EXPLICIT:
             self.shape_pairs_max = min(self.shape_pairs_max, len(shape_pairs_filtered))
 
         # Initialize broad phase
         if self.broad_phase_mode == BroadPhaseMode.NXN: