import Route from '@ember/routing/route';
import { scheduleOnce } from '@ember/runloop';
import { inject } from '@ember/service';

export default Route.extend({
    codePrettify: inject(),
    init() {
        this._super(...arguments);
        
        scheduleOnce('afterRender', this, function() {
            this.get('codePrettify').prettify();
        });
    }
});
