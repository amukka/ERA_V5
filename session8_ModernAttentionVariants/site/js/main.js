/* Widget registry. Each widget module self-registers on import; adding the next
   one is a single import line plus a matching container div in index.html. */

import { mountAll } from './lib.js';

import './widgets/s02-walkthrough.js';
import './widgets/s03-bills.js';
import './widgets/s04-softmax-off.js';
import './widgets/s05-add-only-state.js';
import './widgets/s06-delta-rule.js';
import './widgets/s07-topk.js';
import './widgets/s08-rope.js';
import './widgets/s09-drope.js';
import './widgets/s10-cache-bill.js';
import './widgets/s11-gqa.js';
import './widgets/s12-compression.js';
import './widgets/s13-schedule.js';
import './widgets/s14-memory-stream.js';
import './widgets/s15-readiness.js';
import './widgets/s16-v5-board.js';

mountAll();
