import Route from '@ember/routing/route';

export default Route.extend({
  model(params) {
    return fetch("/assets/" + params.plan_id + ".json")
      .then(function(res) {
        return res.json()
      }
    )
  }
});
