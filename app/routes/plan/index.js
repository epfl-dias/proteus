import Route from '@ember/routing/route';
import { json } from 'd3-fetch'

export default Route.extend({
    model(){
        return json("/assets/plan.json");
    }
});
