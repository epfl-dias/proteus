import Route from '@ember/routing/route';
import { json } from 'd3-fetch'

export default Route.extend({
    actions: {
        echoTest() {
            console.log("asd");
        }
    },
    model(){
        return json("assets/flare.json");
    }
});
